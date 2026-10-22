# src/segmentation_transformer/models/transformer.py
import torch.nn as nn
from .attention import MultiHeadSelfAttention

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, d_k, d_v, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), 
            nn.Dropout(dropout))
        
    def forward(self, X):
        X = X + self.attention(self.norm1(X))
        X = X + self.ffn(self.norm2(X))
        return X
    
