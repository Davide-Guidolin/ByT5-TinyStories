from data import TinyStoriesDataset, PadCollator, TinyStories
from t5 import T5
from config import get_config, DataConfig
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
import numpy as np
import os
import time
import math
from dotenv import load_dotenv
import wandb
import random
import argparse

def init_torch_and_random(seed: int = 42):
    device = "cpu"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed_all(seed)
        
    print(f"Using device: {device}")
    
    # use tf32 if available
    torch.set_float32_matmul_precision("high")
    
    return device

def init_wandb(t5_config, data_config, train_config):
    load_dotenv(dotenv_path="./wandb.env")
    
    wandbteam_name = os.getenv("WANDB_TEAM_NAME")
    wandb_project_name = os.getenv("WANDB_PROJECT_NAME")
    
    full_config = {
        "model": t5_config.__dict__,
        "data": data_config.__dict__,
        "training": train_config.__dict__
    }
    run = wandb.init(
        entity=wandbteam_name,
        project=wandb_project_name,
        config=full_config,
    )
    
    return run

def cycle(iter):
    while True:
        for x in iter:
            yield x

# cosine decaying lr with linear warmup from gpt2 paper
def get_lr(step, train_config):
    # linear warmup
    if step < train_config.warmup_steps:
        return train_config.max_lr * (step+1) / train_config.warmup_steps
    if step > train_config.max_step:
        return train_config.min_lr
    
    decay_ratio = (step - train_config.warmup_steps) / (train_config.max_step - train_config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return train_config.min_lr + coeff * (train_config.max_lr - train_config.min_lr)

def generate_story(
    model: T5, 
    prompt: str, 
    data_config: DataConfig, 
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None
) -> str:
    prompt = list(prompt.encode("utf-8"))
    device = next(model.parameters()).device
    
    # create input tensor of bytes
    prompt = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    
    # generate out tensor of bytes
    out = model.generate(
        prompt, 
        max_new_tokens=max_new_tokens, 
        eos_token_id=data_config.eos_token_id, 
        pad_token_id=data_config.pad_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )    
    out = out.squeeze().cpu().tolist()
    
    if isinstance(out, int):
        out = [out]
        
    # filter PAD and EOS
    final_string_parts = []
    byte_chunk = []
    for tok_id in out:
        if tok_id < 256:
            byte_chunk.append(tok_id)
        else:
            # first decode valid bytes
            if byte_chunk:
                final_string_parts.append(bytes(byte_chunk).decode("utf-8", errors='replace'))
                byte_chunk = []

            # append eos or pad
            if tok_id == data_config.eos_token_id:
                final_string_parts.append("[EOS]")
            if tok_id == data_config.pad_token_id:
                final_string_parts.append("[PAD]")
    
    # decode remaining bytes
    if byte_chunk:
        final_string_parts.append(bytes(byte_chunk).decode("utf-8", errors='replace'))
            
    return "".join(final_string_parts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, default="small", help="Model Size, e.g. small, base, large")
    args = parser.parse_args()
    
    t5_config, data_config, train_config, inference_config = get_config(model_size=args.model_size)
    
    device = init_torch_and_random(seed=train_config.random_seed)
    
    if train_config.wandb_log:
        run = init_wandb(t5_config, data_config, train_config)
        generation_table = wandb.Table(columns=["step", "prompt", "generation"], log_mode="INCREMENTAL")
    
    os.makedirs(train_config.checkpoint_folder, exist_ok=True)
    
    
    # load dataset
    dataset = TinyStories()
    
    train_set = TinyStoriesDataset(
        data_config, 
        ts_dataset=dataset,
        split="train",
        block_size=t5_config.block_size
    )
    train_loader = DataLoader(
        train_set, 
        batch_size=train_config.B,
        collate_fn=PadCollator(data_config.pad_token_id),
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    train_iter = cycle(train_loader)
        
    valid_set = TinyStoriesDataset(
        data_config, 
        ts_dataset=dataset,
        split="validation",
        block_size=t5_config.block_size
    )
    valid_loader = DataLoader(
        valid_set, 
        batch_size=train_config.B,
        collate_fn=PadCollator(data_config.pad_token_id),
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    # create model
    model = T5(t5_config)
    model.print_info()
    
    model.to(device)
    model.to(torch.bfloat16)
    model.train()
    
    
    if device == "cuda":
        cuda_cap = torch.cuda.get_device_capability()
        if cuda_cap[0] >= 7:
            model = torch.compile(model, mode="default", fullgraph=True)
        else:
            print(f"Cannot compile the model. Cuda capability {cuda_cap[0]}.{cuda_cap[1]} < 7.0")
    
    # optimizer
    lr = get_lr(0, train_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    optimizer.zero_grad()

    accumulated_loss = 0.0
    accumulated_delta_t = 0.0
    accumulated_tok_per_sec = 0.0
    
    # training loop
    gradient_step = 0
    for step in range(train_config.max_step):
        torch.compiler.cudagraph_mark_step_begin()
        t0 = time.perf_counter()
        source, dec_input, target = next(train_iter)
        
        source = source.to(device)
        dec_input = dec_input.to(device)
        target = target.to(device)
        
        with autocast(dtype=torch.bfloat16, device_type=device):
            logits = model(source, dec_input)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=data_config.pad_token_id
            )
        
        accumulated_loss += loss.item()
        
        loss = loss / train_config.accumulation_steps
        loss.backward()
        
        # optimize and log
        if (step + 1) % train_config.accumulation_steps == 0:
            # clip gradients
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = get_lr(step, train_config)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            optimizer.step()
            optimizer.zero_grad()
            
            avg_loss = accumulated_loss / train_config.accumulation_steps
            accumulated_loss = 0.0
            avg_dt = accumulated_delta_t / train_config.accumulation_steps
            accumulated_delta_t = 0.0
            avg_tok_per_sec = accumulated_tok_per_sec / train_config.accumulation_steps
            accumulated_tok_per_sec = 0.0
            
            if train_config.wandb_log:                
                run.log({
                    "gradient_step": gradient_step,
                    "train_loss": avg_loss,
                    "learning_rate": lr,
                    "grad_norm": norm,
                    "dt": avg_dt,
                    "tok_per_sec": avg_tok_per_sec
                }, step=step+1)
            
            print(f"gradient_step {gradient_step+1:5d} | step {step+1:5d} | loss: {avg_loss:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {avg_dt:.2f} ms | tok/sec: {avg_tok_per_sec:.2f} ")
            
            gradient_step += 1
        
        # log generation
        if (step + 1) % train_config.log_generation_step == 0:
            model.eval()
            model_out = generate_story(
                model, 
                prompt=inference_config.prompt_text, 
                data_config=data_config, 
                max_new_tokens=t5_config.block_size,
                temperature=inference_config.temperature,
                top_k=inference_config.top_k,
                top_p=inference_config.top_p
            )
            model.train()
            print(f"{inference_config.prompt_text} {model_out}")
            
            if train_config.wandb_log:
                generation_table.add_data(step + 1, inference_config.prompt_text, model_out)
                
                run.log({
                    "generation": generation_table
                }, step=step+1)
            
        # save checkpoint
        if (step + 1) % train_config.save_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'gradient_step': gradient_step,
                'loss': loss.item()
            }
            print(f"Saving checkpoint at step {step}...")
            torch.save(checkpoint, os.path.join(train_config.checkpoint_folder, f'checkpoint_{step}.pt'))
            print("Checkpoint saved")
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000
        tok_per_sec = source.numel() / (t1 - t0)
        
        accumulated_delta_t += dt
        accumulated_tok_per_sec += tok_per_sec