# src/segformer/models/attention.py
import torch.nn as nn
import math
import torch

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.q_proj = nn.Linear(in_features=d_model, out_features= num_heads* d_k, bias=False)
        self.k_proj = nn.Linear(in_features=d_model, out_features= num_heads* d_k, bias=False)
        self.v_proj = nn.Linear(in_features=d_model, out_features= num_heads* d_v, bias=False)
        self.out_proj = nn.Linear(in_features=num_heads* d_v, out_features=d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, X):
        # B - Batch
        # N - Number of tokens (in this case, it will be number of grid H/patch_size * W/patch_size)
        B, N, _ = X.shape
        """
        After q_proj, Q will be (B, N, num_heads*dk)
        so after reshape it becomes (B, N, num_heads, dk)
        and after transpose it becomes (B, num_heads, N, dk ).. 
        so it's equivalent of num_heads matrices of Nxdk shape! and that's how we have split this into multiple heads!
        """
        Q = self.q_proj(X).reshape(B, N, self.num_heads, self.d_k).transpose(1,2)
        K = self.k_proj(X).reshape(B, N, self.num_heads, self.d_k).transpose(1,2)
        V = self.v_proj(X).reshape(B, N, self.num_heads, self.d_v).transpose(1,2)
        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(self.d_k)
        assert scores.shape == (B, self.num_heads, N, N)
        # dim -1 because we want it across columns, i.e. Keys
        attention = torch.softmax(scores, dim=-1)
        attention = self.attn_drop(attention)
        Z = attention @ V # shape (B, heads, N, d_v)
        Z = Z.transpose(1,2).contiguous().reshape(B, N, self.num_heads * self.d_v)
        Z = self.out_proj(Z)
        return self.proj_drop(Z)
