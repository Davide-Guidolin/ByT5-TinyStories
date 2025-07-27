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
    
class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        block_size: int,
        n_embd: int,
        n_head: int,
        attn_droput: float = 0.0,
        attn_proj_droput: float = 0.0     
    ):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.block_size = block_size
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
        
        mask = torch.tril(torch.ones(self.block_size, self.block_size)).view(1, 1, self.block_size, self.block_size)
        self.register_buffer("bias", mask)
        
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
        # mask future tokens
        mask = self.bias[:, :, :T, :T] == 0
        attn = attn.masked_fill(mask, -torch.inf)
        # softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v # (B, n_head, T, dim_head)
        
        # output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.proj_dropout(y)
        
        return y

class MLP(nn.Module):
    """
    FFN with GEGLU. See https://arxiv.org/pdf/2002.05202
    GEGLU: h_gate = x @ W_gate h_up = x @ W_up -> h = gelu(h_gate) * h_up
            -> y = h @ W_down
    """
    def __init__(
        self,
        n_embd: int,
        hidden_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.w_gate = nn.Linear(n_embd, hidden_size, bias=False)
        self.w_up = nn.Linear(n_embd, hidden_size, bias=False)
        self.w_down = nn.Linear(hidden_size, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h_gate = self.w_gate(x) # (B, T, hidden_size)
        h_up = self.w_up(x) # (B, T, hidden_size)
        h = F.gelu(h_gate, approximate='tanh') * h_up # (B, T, hidden_size)
        
        y = self.w_down(h) # (B, T, n_embd)
        y = self.dropout(y)
        
        return y

class EncoderBlock(nn.Module):
    def __init__(
        self,
        config: T5Config
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=False)
        self.attn = MultiHeadSelfAttention(
            config.n_embd, 
            config.n_head, 
            config.attn_dropout,
            config.attn_proj_dropout
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config.n_embd, config.mlp_hidden_size, config.mlp_dropout)
        self.dropout = nn.Dropout(config.skip_conn_dropout)
        
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln_1(x)))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        
        return x        
        
            
class T5(nn.Module):
    
    def __init__(self, config: T5Config):
        super().__init__()
        
        self.n_head = config.n_head
        self.n_layer = config.n_layer
        

if __name__ == "__main__":
    
    # check if it runs
    block = EncoderBlock(T5Config())
    
    x = torch.randn(1, 10, T5Config.n_embd)    
    x = block(x)
    