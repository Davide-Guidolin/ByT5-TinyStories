from dataclasses import dataclass

@dataclass
class T5Config:
    block_size: int = 256 # context length
    n_embd: int = 256 # embedding size
    n_head: int = 4 # number of attention heads
    n_layer: int = 4 # number of layers