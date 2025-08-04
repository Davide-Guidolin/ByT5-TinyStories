from dataclasses import dataclass

@dataclass
class T5Config:
    vocab_size: int = 258 # vocabulary size. 0 - 255 bytes, 256 PAD, 257 EOS
    block_size: int = 1024 # max context length
    n_embd: int = 1472 # embedding size
    n_head: int = 4 # number of attention heads
    n_layer_dec: int = 4 # number of decoder layers
    n_layer_enc: int = 3 * n_layer_dec # number of encoder layers
    
    n_position_buckets: int = 32 # number of embeddings for relative positional encoding
    max_bucket_offset: int = 128 # max offset used to assign a bucket
    mlp_hidden_size: int = 3584 # mlp hidden size
    
    attn_dropout: float = 0.1 # attention dropout
    attn_proj_dropout: float = 0.1 # attention dropout
    mlp_dropout: float = 0.1 # mlp dropout
    skip_conn_dropout: float = 0.1 # dropout on the skip connection
    embd_dropout: float = 0.1 # dropout on the embedding
    
@dataclass
class DataConfig:
    pad_token_id: int = 256
    eos_token_id: int = 257
    mask_pct: float = 0.15 # % of token to mask
    mean_span_corruption_length: int = 20 # mean span corruption length
    
@dataclass
class TrainConfig:
    B: int = 128 # batch size
    max_lr: float = 6e-4
    min_rl: float = max_lr * 0.1
    warmup_step: int = 10
    max_step: int = 50