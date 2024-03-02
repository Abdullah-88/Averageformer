import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from aft_pytorch import AFTFull



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class AFTBlock(nn.Module):
    def __init__(self,dim,dim_ffn, dropout):
        super().__init__()     
        self.AFT = AFTFull(
				    max_seqlen=512,
				    dim=dim,
				    hidden_dim=dim
				)
        self.norm = nn.LayerNorm(dim)       
        
        self.ffn = FeedForward(dim,dim_ffn,dropout)

    def forward(self, x):
        res = x  
        x = self.norm(x) 
        x = self.AFT(x)
        x = res + x
        res = x
        x = self.norm(x)
        x = self.ffn(x)
        out = x + res
        return out     



class AFTGatingUnit(nn.Module):
    def __init__(self,d_model,d_ffn,dropout):
        super().__init__()
        self.aft_1 = AFTBlock(d_model,d_ffn,dropout)
        self.aft_2 = AFTBlock(d_model,d_ffn,dropout)
	
       

    def forward(self, x):
        u, v = x, x 
        u = self.aft_1(u)  
        v = self.aft_2(v)
        out = u * v
        return out


class AverageBlock(nn.Module):
    def __init__(self, d_model, d_ffn,dropout):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.fgu = AFTGatingUnit(d_model,d_ffn,dropout)
        self.ffn = FeedForward(d_model,d_ffn,dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fgu(x)   
        x = x + residual      
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        out = x + residual
        return out



class Averageformer(nn.Module):
    def __init__(self, d_model, d_ffn, num_layers,dropout):
        super().__init__()
        
        self.model = nn.Sequential(
            *[AverageBlock(d_model,d_ffn,dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








