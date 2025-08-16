import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from data import TinyStoriesDataset, PadCollator, TinyStories
from t5 import T5
from config import T5Config, DataConfig, TrainConfig
import time

def init_torch(seed: int = 42):
    device = "cpu"
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(seed)
        
    print(f"Using device: {device}")
    
    # use tf32 if available
    torch.set_float32_matmul_precision("high")
    
    return device

def cycle(iter):
    while True:
        for x in iter:
            yield x

if __name__ == "__main__":
    device = init_torch()
    
    t5_config = T5Config()
    data_config = DataConfig()
    train_config = TrainConfig()
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.max_lr, betas=(0.9, 0.95), eps=1e-8)
    optimizer.zero_grad()

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
        
        loss = loss / train_config.accumulation_step
        
        loss.backward()
        
        if (step + 1) % train_config.accumulation_step == 0:
            # clip gradients
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            print(f"step {step:4d} | grad_norm: {norm:.4f}")
            # lr = get_lr(step)
            optimizer.step()
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000
        tok_per_sec = source.numel() / (t1 - t0)
        
        print(f"step {step:4d} | loss: {loss.item():.6f} | lr: TODO | dt: {dt:.2f}ms | tok/sec: {tok_per_sec:.2f}")