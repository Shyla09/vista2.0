import torch
import torch.nn as nn
import math

class MultiheadCrossAttention(nn.Module):
    """
    Multihead Cross-Attention where object tokens (Queries) attend to text tokens (Keys/Values).
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None):
        B, N, D = query.shape
        _, S, _ = key.shape
        
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
            attn_logits = attn_logits.masked_fill(mask == 0.0, -1e4)
            
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(context)
        
        return out

class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention Layer combining Multihead Cross-Attention and FFN.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiheadCrossAttention(embed_dim, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, memory, memory_key_padding_mask=None):
        attn_out = self.cross_attn(self.norm1(x), memory, memory, memory_key_padding_mask)
        x = x + self.dropout1(attn_out)
        
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout2(mlp_out)
        
        return x
