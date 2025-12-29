
'''
Copyright (c) 2021, Alibaba Cloud and its affiliates;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''



import torch.nn as nn
import torch.nn.functional as F
import torch

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim):
        """
        Rotary Position Embedding (RoPE)
        
        参数:
            dim (int): 输入特征的维度（必须是偶数）
        """
        super().__init__()
        assert dim % 2 == 0, "维度必须是偶数"
        self.dim = dim
        
        # 初始化频率参数 theta_i = 10000^(-2i/d)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 缓存cos和sin值
        self.cos_cached = None
        self.sin_cached = None
        self.max_seq_len = None

    def _update_cos_sin_cache(self, seq_len, device):
        """更新cos和sin缓存"""
        if self.max_seq_len is None or seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            # 生成位置索引 [0, 1, ..., seq_len-1]
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            
            # 计算频率矩阵 shape: (seq_len, dim//2)
            freqs = torch.outer(t, self.inv_freq)
            
            # 拼接相同频率用于后续广播 shape: (seq_len, dim)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # 缓存cos和sin值
            self.cos_cached = emb.cos()[None, :, None, :]  # 形状: (1, seq_len, 1, dim)
            self.sin_cached = emb.sin()[None, :, None, :]

    def rotate_half(self, x):
        """将输入的后半部分旋转并取反"""
        x = x.view(*x.shape[:-1], -1, 2)        # 分组成二维向量
        x1 = x[..., 0]                          # 前半部分
        x2 = x[..., 1]                          # 后半部分
        x_rot = torch.stack((-x2, x1), dim=-1)   # 旋转操作
        return x_rot.view(*x.shape[:-2], -1)     # 恢复原始形状

    def forward(self, x, offset=0):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量，形状为 (batch, seq_len, n_head, dim)
            offset (int): 位置偏移量（用于增量解码）
        
        返回:
            Tensor: 旋转后的张量
        """
        batch, seq_len, n_head, _ = x.size()
        self._update_cos_sin_cache(seq_len + offset, x.device)
        
        # 获取对应位置的cos和sin值
        cos = self.cos_cached[:, offset:offset+seq_len]
        sin = self.sin_cached[:, offset:offset+seq_len]
        
        # 应用旋转位置编码
        x_rot = (x * cos) + (self.rotate_half(x) * sin)
        return x_rot
