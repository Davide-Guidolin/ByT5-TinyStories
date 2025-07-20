from dataclasses import dataclass

@dataclass
class TransformerConfig:
    block_size: int = 256 # context length
    n_embd: int = 256 # embedding size
    n_head: int = 4 # number of attention heads
    n_layer: int = 4 # number of layers