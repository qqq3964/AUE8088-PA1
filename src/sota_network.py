import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import math

"""
https://arxiv.org/pdf/2010.11929v2
ICLR 2021 sota paper ViT is vision transformer
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.heads = heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.d_k = dim ** 0.5  
        
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.query(x) # (B, N, D)
        K = self.key(x) # (B, N, D)
        V = self.value(x) # (B, N, D)
        
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.heads)
        
        output = (Q @ rearrange(K, 'b h n d -> b h d n')) / self.d_k  # (B, N, N)
        output = F.softmax(output, dim=-1)
        output = output @ V         
        output = rearrange(output, 'b h n d -> b n (h d)', h=self.heads)
            
        output = self.proj(output)
        output = x + output
        output = self.norm(output)
        
        return output
    
class FeedForward(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        output = self.linear1(x)
        output = self.relu(output)
        output = self.linear2(output)
        output = x + output
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, dim=512, max_len=5000):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000) / dim)).unsqueeze(0)

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.shape[1], :].unsqueeze(0)
        x = self.dropout(x)
        
        return x 
    
class ViT(nn.Module):
    def __init__(self, 
                 num_classes=200,
                 patch=8, 
                 dim=512,
                 max_len=5000,
                 heads=8,
                 N=6):
        super().__init__()  
        
        self.num_classes = num_classes
        self.patch = patch
        
        self.flatten_layer = nn.Linear(patch * patch * 3, dim) # 3 channel input
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        
        layers = [
            PositionalEncoding(dim=dim, max_len=max_len)
        ]        
        for _ in range(N):
            layers.append(MultiHeadAttention(dim=dim, heads=heads))
            layers.append(FeedForward(dim=dim))
        self.encoder = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim)
        
        # classification
        self.classifier = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        assert H % self.patch == 0 and W % self.patch == 0

        x = rearrange(x, 'b (h ph) (w pw) c -> b (h w) (ph pw c)', ph=self.patch, pw=self.patch)
        
        bert_token = self.class_token.expand(B, 1, -1)
        x = self.flatten_layer(x)
        x = torch.cat([bert_token, x], dim=1)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        output = self.classifier(cls_token)
        
        return output 