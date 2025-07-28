from dataclasses import dataclass

@dataclass
class T5Config:
    vocab_size: int = 100 # vocabulary size
    block_size: int = 1024 # max context length
    n_embd: int = 256 # embedding size
    mlp_hidden_size: int = 4 * n_embd # mlp hidden size
    n_head: int = 4 # number of attention heads
    n_layer_enc: int = 4 # number of encoder layers
    n_layer_dec: int = 4 # number of decoder layers
    
    attn_dropout: float = 0.1 # attention dropout
    attn_proj_dropout: float = 0.1 # attention dropout
    mlp_dropout: float = 0.1 # mlp dropout
    skip_conn_dropout: float = 0.1 # dropout on the skip connection
    embd_dropout: float = 0.1 # dropout on the embedding