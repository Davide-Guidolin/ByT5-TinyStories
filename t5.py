import torch
import torch.nn as nn
from config import T5Config

# https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511

class SelfAttention(nn.Module):

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        droput: float = 0.0     
    ):
        self.n_head = n_head
        self.dim_head = n_embd // n_head
        self.scale = self.dim_head ** -0.5
        
        # key, query, value for all heads (will be split later)
        self.attn = nn.Linear(n_embd, 3*n_embd, bias=False)
        # final projection
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.dropout = nn.Dropout(p=droput)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

        
        
class T5(nn.Module):
    
    def __init__(self, config: T5Config):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_layer = config.n_layer