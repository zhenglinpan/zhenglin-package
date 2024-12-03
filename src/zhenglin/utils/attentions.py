import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    A method from [CBAM: Convolutional Block Attention Module]
    Alternative Implementation: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    """
    def __init__(self, chan_in, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # global pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(chan_in, chan_in // 16, kernel_size, stride, padding, bias=False),
                                 nn.ReLU(1),
                                 nn.Conv2d(chan_in // 16, chan_in, kernel_size, stride, padding, bias=False))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        
        return out
        
class SpacialAttention(nn.Module):
    """
    A method from [CBAM: Convolutional Block Attention Module]
    Alternative Implementation: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    """
    def __init__(self, kernel_size=7, stride=1, padding=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, stride, padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        
        return x
    
class CBAMRes(nn.Module):
    """
        An example of a ResNet basic block with CBAM
    """
    def __init__(self):
        self.res_block = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                       nn.ReLU(1),
                                       nn.BatchNorm2d(64),
                                       nn.Conv2d(64, 64, 3, 1, 1),
                                       nn.BatchNorm2d(64))
        self.shortcut = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                      nn.ReLU(1),
                                      nn.BatchNorm2d(64))
        
        self.ca = ChannelAttention(64)
        self.sa = SpacialAttention()
        self.relu = nn.ReLU(1)

    def forward(self, x):
        out = self.res_block(x)
        
        out = self.ca(out) * out  # channel and spacial attention
        out = self.sa(out) * out
        
        out = self.relu(out + self.shortcut(x))
        return out


class MultiHeadAttention(nn.Module):
    """
    A method from [Attention is All You Need]
    An implementation from: https://github.com/hyunwoongko/transformer/blob/master/README.md
    """
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaleDotProductAttention()
        
    def forward(self, x):
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        
        q, k, v = self.split(q), self.split(k), self.split(v)
        
        out, attention = self.attention(q, k, v)
        
        out = self.concat(out)
        out = self.w_o(out)
        
        return out
        
    def split(self, x):
        batch_size, seq_len, d_model = x.size()
        d_tensor = d_model // self.n_head
        return x.view(batch_size, seq_len, self.n_head, d_tensor).transpose(1, 2)
    
    def concat(self, x):
        batch_size, n_head, seq_len, d_tensor = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, n_head*d_tensor)


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax2d(dim=-1)
    
    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, n_head, seq_len, d_tensor = k.size()
        
        score = (q @ k.transpose(2, 3)) / (d_tensor ** 0.5) # score: seq_len x seq_len
        
        if mask is not None:
            score = score.masked_fill(mask==0, -10000)
            
        score = self.softmax(score) 
        
        v = score @ v   # the same shape as if intact
        
        return v, score