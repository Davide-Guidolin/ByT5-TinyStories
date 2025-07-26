from dataclasses import dataclass

@dataclass
class T5Config:
    block_size: int = 256 # context length
    n_embd: int = 256 # embedding size
    mlp_hidden_size: int = 4 * n_embd # mlp hidden size
    n_head: int = 4 # number of attention heads
    n_layer: int = 4 # number of layers
    
    attn_dropout: float = 0.1 # attention dropout
    attn_proj_dropout: float = 0.1 # attention dropout
    mlp_dropout: float = 0.1 # mlp dropout
    skip_conn_dropout: float = 0.1 # dropout on the skip connection