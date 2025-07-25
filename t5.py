import torch
import torch.nn as nn
from torch.nn import functional as F
from config import T5Config

# https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_droput: float = 0.0,
        attn_proj_droput: float = 0.0     
    ):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.dim_head = n_embd // n_head
        self.scale = self.dim_head ** -0.5
        
        # key, query, value for all heads (will be split later)
        self.attn = nn.Linear(n_embd, 3*n_embd, bias=False)
        # final projection
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(p=attn_droput)
        self.proj_dropout = nn.Dropout(p=attn_proj_droput)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()
        
        qkv = self.attn(x) # (B, T, C * 3)
        # split in 3
        q, k, v = qkv.split(self.n_embd, dim=2) # 3 x (B, T, C)
        
        # prepare for multi-head
        q = q.view(B, T, self.n_head, self.dim_head).transpose(1, 2) # (B, n_head, T, dim_head)
        k = k.view(B, T, self.n_head, self.dim_head).transpose(1, 2) # (B, n_head, T, dim_head)
        v = v.view(B, T, self.n_head, self.dim_head).transpose(1, 2) # (B, n_head, T, dim_head)
        
        # attention calculation
        attn = (q @ k.transpose(-2, -1)) / self.scale # (B, n_head, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v # (B, n_head, T, dim_head)
        
        # output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.proj_dropout(y)
        
        return y
        
        
class T5(nn.Module):
    
    def __init__(self, config: T5Config):
        super().__init__()
        
        self.n_head = config.n_head
        self.n_layer = config.n_layer
        

if __name__ == "__main__":
    
    # check if it runs
    mh_attn = MultiHeadSelfAttention(T5Config.n_embd, T5Config.n_head, attn_droput=0.1, attn_proj_droput=0.1)
    
    x = torch.randn(1, 10, T5Config.n_embd)    
    y = mh_attn(x)