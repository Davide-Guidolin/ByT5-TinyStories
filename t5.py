import torch
import torch.nn as nn
from torch.nn import functional as F
from config import T5Config

# https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        block_size: int,
        n_embd: int,
        n_head: int,
        attn_droput: float = 0.0,
        attn_proj_droput: float = 0.0,
        is_causal: bool = False
    ):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.block_size = block_size
        self.n_head = n_head
        self.n_embd = n_embd
        self.dim_head = n_embd // n_head
        self.is_causal = is_causal
        
        # key, query, value for all heads (will be split later)
        self.attn = nn.Linear(n_embd, 3*n_embd, bias=False)
        # final projection
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(p=attn_droput)
        self.proj_dropout = nn.Dropout(p=attn_proj_droput)
        
        if self.is_causal:
            mask = torch.tril(torch.ones(self.block_size, self.block_size)).view(1, 1, self.block_size, self.block_size)
            self.register_buffer("bias", mask)
        
    def forward(self, x: torch.Tensor, position_bias: torch.Tensor = None) -> torch.Tensor:
        
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
        # attn = (q @ k.transpose(-2, -1)) # (B, n_head, T, T)
        
        # if position_bias is not None:
        #     attn = attn + position_bias
            
        # attn = attn / self.scale
        
        # attn = F.softmax(attn, dim=-1)
        # attn = self.attn_dropout(attn)
        # y = attn @ v # (B, n_head, T, dim_head)
        
        # flash attention
        attn_mask = position_bias
        if self.is_causal:
            # mask future tokens
            causal_mask = self.bias[:, :, :T, :T] == 0
            
            if attn_mask is None:
                attn_mask = torch.zeros(1, 1, T, T, dtype=q.dtype, device=q.device).masked_fill(causal_mask, -torch.inf)
            else:
                attn_mask = attn_mask.masked_fill(causal_mask, -torch.inf)
        
        y = F.scaled_dot_product_attention(
            q, 
            k, 
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        
        # output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.proj_dropout(y)
        
        return y

class CrossAttention(nn.Module):
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
        
        # query, key, value
        self.attn_q = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_kv = nn.Linear(n_embd, 2*n_embd, bias=False)
        # final projection
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(p=attn_droput)
        self.proj_dropout = nn.Dropout(p=attn_proj_droput)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, position_bias: torch.Tensor = None) -> torch.Tensor:
        
        B, T_q, C = x_q.size()
        _, T_kv, _ = x_kv.size()
        
        q = self.attn_q(x_q)
        kv = self.attn_kv(x_kv)
        k, v = kv.split(self.n_embd, dim=2)
        
        q = q.view(B, T_q, self.n_head, self.dim_head).transpose(1, 2)
        k = k.view(B, T_kv, self.n_head, self.dim_head).transpose(1, 2)
        v = v.view(B, T_kv, self.n_head, self.dim_head).transpose(1, 2)
        
        # attn = (q @ k.transpose(-2, -1))
        
        # if position_bias is not None:
        #     attn = attn + position_bias
            
        # attn = attn / self.scale
   
        # attn = F.softmax(attn, dim=-1)
        # attn = self.attn_dropout(attn)
        # y = attn @ v
        
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=position_bias,
            dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            config.block_size,
            config.n_embd, 
            config.n_head, 
            config.attn_dropout,
            config.attn_proj_dropout,
            is_causal=False
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config.n_embd, config.mlp_hidden_size, config.mlp_dropout)
        self.dropout = nn.Dropout(config.skip_conn_dropout)
        
    def forward(self, x: torch.Tensor, position_bias: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln_1(x), position_bias=position_bias))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        
        return x        
        
class DecoderBlock(nn.Module):
    def __init__(
        self,
        config: T5Config
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=False)
        self.causal_attn = MultiHeadSelfAttention(
            config.block_size,
            config.n_embd, 
            config.n_head, 
            config.attn_dropout,
            config.attn_proj_dropout,
            is_causal=True
        )
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=False)
        self.cross_attn = CrossAttention(
            config.block_size,
            config.n_embd, 
            config.n_head, 
            config.attn_dropout,
            config.attn_proj_dropout
        )
        self.ln_3 = nn.LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config.n_embd, config.mlp_hidden_size, config.mlp_dropout)
        self.dropout = nn.Dropout(config.skip_conn_dropout)
        
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, position_bias_self_attn: torch.Tensor = None, position_bias_cross_attn: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.causal_attn(self.ln_1(x), position_bias_self_attn))
        x = x + self.dropout(self.cross_attn(self.ln_2(x), encoder_out, position_bias_cross_attn))
        x = x + self.dropout(self.mlp(self.ln_3(x)))
        
        return x
  
class T5(nn.Module):
    
    def __init__(self, config: T5Config):
        super().__init__()
        
        self.config = config
        
        self.relative_wpe = nn.Embedding(config.n_position_buckets, config.n_head)
        
        self.transformer = nn.ModuleDict(dict(
            # word token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # dropout for embedding
            drop = nn.Dropout(config.embd_dropout),
            # encoder
            encoder = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer_enc)]),
            ln_enc = nn.LayerNorm(config.n_embd, bias=False),
            # decoder
            decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer_dec)]),
            ln_dec = nn.LayerNorm(config.n_embd, bias=False)
        ))
        # Final head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying with embedding
        self.lm_head.weight = self.transformer.wte.weight
        
    def _relative_position_bucket(self, relative_position: torch.Tensor, num_buckets: int = 32, max_distance: int = 128, bidirectional: bool = True) -> torch.Tensor:
        """
        Calculates the bucket index for a given relative position.

        Args:
            relative_position: A tensor of relative positions.
            num_buckets: The total number of buckets.
            max_distance: The maximum distance to consider before bucketing.
            bidirectional: Whether the attention is bidirectional (encoder) or not (decoder).
   
        Returns:
           A tensor of bucket indices.
        """
        res = 0
        
        # negate relative positions -> think about relative position as 'distance into the past' e.g. 2 means the key is 2 step behind the query
        relative_position = -relative_position
        
        if bidirectional:
            # if bidirectional half buckets are used for negative values, half for positive values
            num_buckets = num_buckets // 2
            res += (relative_position < 0).to(torch.long) * num_buckets # adds num_bucket to negative positions, e.g. for 32 bucket 0 - 15 will be for positive positions, 16 - 31 for negative
            relative_position = relative_position.abs()
        else:
            # if not bidirectional use all the available buckets
            relative_position = torch.max(relative_position, torch.zeros_like(relative_position)) # if causal attention, clip negative values (i.e. key ahead query -> future tokens)
            
        # half of available buckets, i.e. original_num_buckets // 4 for bidirectional and original_num_buckets // 2 for non bidirectional
        # is used for direct mapping with absolute value, the other half is used for log scaling
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # log scaling for positions >= max_exact
        val_if_large = max_exact + (torch.log(relative_position.float() / max_exact) / torch.log(torch.tensor(max_distance / max_exact)) * (num_buckets - max_exact)).to(torch.long)
        # make sure bucket index does not exceed the maximum
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        res += torch.where(is_small, relative_position, val_if_large)
        return res
    
    def encode(self, src_idx: torch.Tensor) -> torch.Tensor:
        _, T_src = src_idx.size()
        
        enc_pos = torch.arange(0, T_src, dtype=torch.long, device=src_idx.device)
        enc_relative_position = enc_pos[None, :] - enc_pos[:, None]
        enc_rp_bucket = self._relative_position_bucket(
            enc_relative_position, 
            self.config.n_position_buckets, 
            self.config.max_bucket_offset,
            bidirectional=True
        )
          
        enc_position_bias = self.relative_wpe(enc_rp_bucket) # (T_src, T_src, n_head)
        enc_position_bias = enc_position_bias.permute(2, 0, 1).unsqueeze(0) # (1, n_head, T_src, T_src)
    
        src_emb = self.transformer.wte(src_idx)
        enc_x = self.transformer.drop(src_emb)
        
        for block in self.transformer.encoder:
            enc_x = block(enc_x, position_bias=enc_position_bias)
        
        encoder_out = self.transformer.ln_enc(enc_x)
        
        return encoder_out
    
    def decode(self, trg_idx: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        _, T_trg = trg_idx.size()
        _, T_src, _ = encoder_out.size()
        
        dec_pos = torch.arange(0, T_trg, dtype=torch.long, device=trg_idx.device)
        dec_relative_position = dec_pos[None, :] - dec_pos[:, None]
        dec_rp_bucket = self._relative_position_bucket(
            dec_relative_position, 
            self.config.n_position_buckets, 
            self.config.max_bucket_offset, 
            bidirectional=False
        )
          
        dec_position_bias_self_attn = self.relative_wpe(dec_rp_bucket) # (T_trg, T_trg, n_head)
        dec_position_bias_self_attn = dec_position_bias_self_attn.permute(2, 0, 1).unsqueeze(0) # (1, n_head, T_trg, T_trg)
        
        enc_pos = torch.arange(0, T_src, dtype=torch.long, device=encoder_out.device)
        dec_relative_position_cross_attn = enc_pos[None, :] - dec_pos[:, None]
        dec_rp_bucket_cross_attn = self._relative_position_bucket(
            dec_relative_position_cross_attn, 
            self.config.n_position_buckets, 
            self.config.max_bucket_offset,
            bidirectional=True
        )
        dec_position_bias_cross_attn = self.relative_wpe(dec_rp_bucket_cross_attn)
        dec_position_bias_cross_attn = dec_position_bias_cross_attn.permute(2, 0, 1).unsqueeze(0)
        
        trg_emb = self.transformer.wte(trg_idx)
        dec_x = self.transformer.drop(trg_emb)
        
        for block in self.transformer.decoder:
            dec_x = block(
                dec_x, 
                encoder_out, 
                position_bias_self_attn=dec_position_bias_self_attn, 
                position_bias_cross_attn=dec_position_bias_cross_attn
            )

        decoder_out = self.transformer.ln_dec(dec_x)
        
        # Final Head
        logits = self.lm_head(decoder_out)
        
        return logits
        
    def forward(self, src_idx: torch.Tensor, trg_idx: torch.Tensor = None) -> torch.Tensor:
        _, T_src = src_idx.size()
        _, T_trg = trg_idx.size()
        assert T_src <= self.config.block_size, f"Cannot forward sequence of length {T_src}, maximum block size is {self.config.block_size}"
        assert T_trg <= self.config.block_size, f"Cannot use sequence of length {T_trg} as target, maximum block size is {self.config.block_size}"
        
        # Encoder
        encoder_out = self.encode(src_idx)
        
        # Decoder
        logits = self.decode(trg_idx, encoder_out)
        
        return logits
    
    @torch.no_grad
    def generate(
        self, 
        prompt: torch.Tensor, 
        max_new_tokens: int = 10, 
        eos_token_id: int = 257, 
        pad_token_id: int = 256,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ) -> torch.Tensor:        
        assert max_new_tokens <= self.config.block_size, f"Cannot generate a sequence of length {max_new_tokens}, maximum block size is {self.config.block_size}"
        
        # run encoder once
        encoder_out = self.encode(prompt)
        
        # start decode with pad token
        dec_idx = torch.tensor([[pad_token_id]], dtype=torch.long, device=prompt.device)
        
        # autoregressive loop
        for _ in range(max_new_tokens):
            logits = self.decode(dec_idx, encoder_out)
            
            # logits for last token
            last_tok_logits = logits[:, -1, :]
            last_tok_logits /= temperature
            
            # sample token
            probs = F.softmax(last_tok_logits, dim=-1)
            
            # top_k filtering
            if top_k is not None:
                top_k_probs, _ = torch.topk(probs, k=top_k, dim=-1)
                top_k_probs = top_k_probs[:, [-1]]
                
                mask = probs >= top_k_probs
                probs = torch.where(mask, probs, 0)
                
                # normalize
                probs = probs / torch.sum(probs, dim=-1, keepdim=True)
                
            # top_p filtering
            if top_p is not None:
                sorted_probs, sorted_idxs = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_idxs_to_remove = cum_probs > top_p
                # include the index going above top_p
                sorted_idxs_to_remove[..., 1:] = sorted_idxs_to_remove[..., :-1].clone()
                # safety check to include at least the first index
                sorted_idxs_to_remove[..., 0] = 0
                
                idxs_to_remove = sorted_idxs_to_remove.scatter(dim=-1, index=sorted_idxs, src=sorted_idxs_to_remove)
                probs[idxs_to_remove] = 0
                
                probs = probs / torch.sum(probs, dim=-1, keepdim=True)
                
                
            next_token = torch.multinomial(probs, num_samples=1)
            
            # concat
            dec_idx = torch.cat([dec_idx, next_token], dim=1)
        
            # check eos
            if next_token.item() == eos_token_id:
                break
            
        return dec_idx[:, 1:]        
        
    def print_info(self):
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())/1e6:.2f} M")

if __name__ == "__main__":
    
    # check if it runs
    config = T5Config()
    model = T5(config)
    
    src = torch.randint(0, config.vocab_size, (1, 10)) # Batch size 1, sequence length 10
    trg = torch.randint(0, config.vocab_size, (1, 5))
    
    logits = model(src, trg)
    print(logits.shape)
    