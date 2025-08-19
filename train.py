from data import TinyStoriesDataset, PadCollator, TinyStories
from t5 import T5
from config import T5Config, DataConfig, TrainConfig
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import os
import time
import math
from dotenv import load_dotenv
import wandb
import random

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

if __name__ == "__main__":
    
    t5_config = T5Config()
    data_config = DataConfig()
    train_config = TrainConfig()
    
    device = init_torch_and_random(seed=train_config.random_seed)
    run = init_wandb(t5_config, data_config, train_config)
    
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
        num_workers=2,
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
        num_workers=2,
        pin_memory=True
    )
    
    # create model
    model = T5(t5_config)
    model.to(device)
    model.to(torch.bfloat16)
    
    model.print_info()
    
    if device == "cuda":
        cuda_cap = torch.cuda.get_device_capability()
        if cuda_cap[0] >= 7:
            model = torch.compile(model)
        else:
            print(f"Cannot compile the model. Cuda capability {cuda_cap[0]}.{cuda_cap[1]} < 7.0")
    
    # optimizer
    lr = get_lr(0, train_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    optimizer.zero_grad()

    accumulated_loss = 0.0
    accumulated_delta_t = 0.0
    accumulated_tok_per_sec = 0.0
    
    # training loop
    for step in range(train_config.max_step):
        t0 = time.perf_counter()
        source, dec_input, target = next(train_iter)
        
        source = source.to(device)
        dec_input = dec_input.to(device)
        target = target.to(device)
        
        with autocast(dtype=torch.bfloat16):
            logits = model(source, dec_input)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=data_config.pad_token_id
            )
        
        accumulated_loss += loss.item()
        
        loss = loss / train_config.accumulation_steps
        loss.backward()
        
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
            
            wandb.log({
                "train_loss": avg_loss,
                "learning_rate": lr,
                "grad_norm": norm,
                "dt": avg_dt,
                "tok_per_sec": avg_tok_per_sec
            }, step=step)
            
            print(f"step {step+1:5d} | loss: {avg_loss:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {avg_dt:.2f} ms | tok/sec: {avg_tok_per_sec:.2f} ")
            
            if (step + 1) % train_config.save_interval == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'loss': loss.item()
                }
                print(f"Saving checkpoint at step {step}...")
                torch.save(checkpoint, f'checkpoint_{step}.pt')
                print("Checkpoint saved")
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000
        tok_per_sec = source.numel() / (t1 - t0)
        
        accumulated_delta_t += dt
        accumulated_tok_per_sec += tok_per_sec