
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import numpy as np
from PositionalEncoding import RotaryPositionEmbedding
from einops import rearrange, repeat
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
'''
* is it feasible to improve the generation quality by using both local and global rotations?
* directly predicting the poses can result in jittering between two inference chuncks.
    is it possible to predict incremental rotations instead of absolute rotations?
* can we generate both audio and gestures together from text? this can reduce the latency of the system. 
* can we improve the generation quality by using a SKEL, which is a biomechanics model
* can bone direction loss improve the generation quality?
* can we use similarity loss on latent structure to improve the generation quality?
* is it feasible to improve the performance by using bone direction loss?
* in the future i shall consider using latent diffusion models to generate the poses.
* i shall try intermediate supervision to improve the generation quality.
* decompose the parameter space into three parts: expression, hand joints, and other body joints.
'''



class MyMultiheadAttention(nn.Module):
    '''
    A multi-head attention module that supports different input dimensions for queries, keys, and values.'''
    def __init__(self, embed_dim, 
                 num_heads, 
                 dropout=0.1,
                 attention_dim=None,
                 qdim=None, kdim=None, vdim=None, out_dim=None, batch_first=False, device=None, dtype=None) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.qdim = qdim if qdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.out_dim = out_dim if out_dim is not None else embed_dim
        self.attention_dim = attention_dim or embed_dim


        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = self.attention_dim // num_heads
        assert self.head_dim * num_heads == self.attention_dim, "embed_dim must be divisible by num_heads"


        self.q_proj_weight = Parameter(torch.empty((self.attention_dim, self.qdim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((self.attention_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))



        self.in_proj_bias_q = Parameter(torch.empty(self.attention_dim, **factory_kwargs))
        self.in_proj_bias_k = Parameter(torch.empty(self.attention_dim, **factory_kwargs))
        self.in_proj_bias_v = Parameter(torch.empty(embed_dim, **factory_kwargs))

        self.out_proj = nn.Linear(embed_dim, self.out_dim, bias=True, **factory_kwargs)


        self._reset_parameters()

    def _reset_parameters(self):

        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)


        constant_(self.in_proj_bias_q, 0.)
        constant_(self.in_proj_bias_k, 0.)
        constant_(self.in_proj_bias_v, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v):
        q = torch.einsum("blj,ij->bli", q, self.q_proj_weight)
        k = torch.einsum("blj,ij->bli", k, self.k_proj_weight)
        v = torch.einsum("blj,ij->bli", v, self.v_proj_weight)

        B, L = q.shape[:2]
        q =  q + self.in_proj_bias_q
        k = k + self.in_proj_bias_k
        v = v + self.in_proj_bias_v

        q = q.reshape((B, L, self.num_heads, -1)).transpose(-3, -2)
        k = k.reshape((B, L, self.num_heads, -1)).transpose(-3, -2)
        v = v.reshape((B, L, self.num_heads, -1)).transpose(-3, -2)
        res = F.scaled_dot_product_attention(q, k,v, dropout_p=self.dropout).transpose(-3, -2)
        res = res.reshape((B, L, -1))
        res = self.out_proj(res)
        return res


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        集成Rotary Position Embedding的多头注意力机制

        参数:
            embed_dim (int): 输入嵌入维度
            num_heads (int): 注意力头数
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 初始化QKV投影矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 初始化Rotary位置编码
        self.rotary_pe = RotaryPositionEmbedding(self.head_dim)

        # 最终输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, offset=0):
        """
        前向传播

        参数:
            query: (batch_size, seq_len, embed_dim)
            key: (batch_size, seq_len, embed_dim)
            value: (batch_size, seq_len, embed_dim)
            key_padding_mask: (batch_size, seq_len) 可选
            attn_mask: (seq_len, seq_len) 可选
            offset: 位置偏移（用于增量解码）

        返回:
            attn_output: (batch_size, seq_len, embed_dim)
        """
        batch_size = query.size(0)
        seq_len = query.size(1)

        # 投影到Q, K, V
        q = self.q_proj(query)  # (B, T, E)
        k = self.k_proj(key)    # (B, T, E)
        v = self.v_proj(value)  # (B, T, E)

        # 分割多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, D)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, D)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, T, D)

        # 应用Rotary位置编码到Q和K
        q = self.rotary_pe(q.permute(0, 2, 1, 3), offset=offset).permute(0, 2, 1, 3)  # 保持形状(B, H, T, D)
        k = self.rotary_pe(k.permute(0, 2, 1, 3), offset=offset).permute(0, 2, 1, 3)

        # 计算注意力分数 (B, H, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 处理掩码
        if key_padding_mask is not None:
            mask = key_padding_mask.view(batch_size, 1, 1, seq_len)  # (B, 1, 1, T)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        if attn_mask is not None:
            attn_scores += attn_mask

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 应用注意力权重到V
        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D)

        # 合并多头
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (B, T, H, D)
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)  # (B, T, E)

        # 最终投影
        attn_output = self.out_proj(attn_output)
        return attn_output


class CrossAttentionLayerWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.ln3 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.ln4 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.attention = MultiHeadAttentionWithRoPE(embed_dim, num_heads)
    
    def forward(self, x, key, value, key_padding_mask=None, attn_mask=None, offset=0):
        x = self.ln1(x)
        key = self.ln2(key)
        value = self.ln3(value)
        attn_output = self.attention(x, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask, offset=offset)
        x = x + attn_output
        x = self.ln4(x)
        x = x + self.ff(x)
        return x


class StyleEmbedding(nn.Module):
    def __init__(self, num_persons, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_persons, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        return self.mlp(t)

class SimpleSpeechModel(nn.Module):
    def __init__(self, 
        feature_dim=512, 
        output_feat_dim=100,
        num_person=31):
        super(SimpleSpeechModel, self).__init__()
        self.num_person = num_person
        self.pos_enc = Summer(PositionalEncoding1D(feature_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, 
                                                   nhead=4,
                                                   dim_feedforward=2 * feature_dim, 
                                                   batch_first=True, 
                                                   norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.audio_feature_map = nn.Linear(1024 + 64, feature_dim)
        self.vertice_map_r = nn.Linear(feature_dim, output_feat_dim)
        self.output_feat_dim = output_feat_dim

        self.style_embed = StyleEmbedding(num_persons=31, dim=64)

        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, huberts, pid):
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(huberts.device)
        style_embed = self.style_embed(one_hots).unsqueeze(1).expand(-1, huberts.shape[1], -1)
        
        vertice_input = self.audio_feature_map(torch.cat([huberts, style_embed], dim=-1))
        vertice_input = self.pos_enc(vertice_input)
        vertice_out = self.transformer_decoder(vertice_input, vertice_input)
        vertice_out = self.vertice_map_r(vertice_out)
        return vertice_out


class SimpleSpeechModelWav2Sem(nn.Module):
    def __init__(self, 
        feature_dim=512, 
        output_feat_dim=100,
        num_person=31):
        super(SimpleSpeechModelWav2Sem, self).__init__()
        self.num_person = num_person
        self.pos_enc = Summer(PositionalEncoding1D(feature_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, 
                                                   nhead=4,
                                                   dim_feedforward=2 * feature_dim, 
                                                   batch_first=True, 
                                                   norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.audio_feature_map = nn.Linear(1024 + 64 + 1024, feature_dim)
        self.vertice_map_r = nn.Linear(feature_dim, output_feat_dim)
        self.output_feat_dim = output_feat_dim

        self.style_embed = StyleEmbedding(num_persons=31, dim=64)

        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, huberts, pid, wav2sem):
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(huberts.device)
        style_embed = self.style_embed(one_hots).unsqueeze(1).expand(-1, huberts.shape[1], -1)
        
        wav2sem = wav2sem.unsqueeze(1).expand(-1, huberts.shape[1], -1)
        vertice_input = self.audio_feature_map(torch.cat([huberts, style_embed, wav2sem], dim=-1))
        vertice_input = self.pos_enc(vertice_input)
        vertice_out = self.transformer_decoder(vertice_input, vertice_input)
        vertice_out = self.vertice_map_r(vertice_out)
        return vertice_out





class DiTBlock(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, x, **kwargs):
        time_embed = kwargs['time_embed']
        x0 = x
        x = self.ln1(x)
        gate1, scale1, shift1, gate2, scale2, shift2 = self.mlp(time_embed).chunk(6, dim=-1)
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0

        x0 = x
        x = self.ln2(x)
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        x = self.ln3(x)
        return x


class DiTBlock2(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln3 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 9)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

        self.ln_seed_poses = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_seed_poses = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.n_seed = n_seed



    def forward(self, x, **kwargs):
        time_emb = kwargs['time_embed']
        seed_poses = kwargs['seed_poses']

        gate1, scale1, shift1, gate2, scale2, shift2, gate3, scale3, shift3 = self.mlp(
            time_emb).chunk(9, dim=-1)
        # seed_poses
        seed_poses0 = seed_poses
        seed_poses = self.ln_seed_poses(seed_poses)
        seed_poses = seed_poses * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        seed_poses = self.ff_seed_poses(seed_poses) * gate3.unsqueeze(1) + seed_poses0
        x_seed_parts = x[:, :self.n_seed]
        x_seed_parts = self.cross_attn(x_seed_parts, seed_poses, seed_poses)[0] + x_seed_parts
        x = torch.cat([x_seed_parts, x[:, self.n_seed:]], dim=1)



        # time_style_embed
        x0 = x
        x = self.ln1(x)
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0

        # ff
        x0 = x
        x = self.ln2(x)
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        
        x = self.ln3(x)
        return x

class DiTBlock2PostNorm(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        # self.ln3 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 9)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

        self.ln_seed_poses = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_seed_poses = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.n_seed = n_seed



    def forward(self, x, **kwargs):
        time_emb = kwargs['time_embed']
        seed_poses = kwargs['seed_poses']

        gate1, scale1, shift1, gate2, scale2, shift2, gate3, scale3, shift3 = self.mlp(
            time_emb).chunk(9, dim=-1)
        # seed_poses
        seed_poses0 = seed_poses
        seed_poses = self.ln_seed_poses(seed_poses)
        seed_poses = seed_poses * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        seed_poses = self.ff_seed_poses(seed_poses) * gate3.unsqueeze(1) + seed_poses0
        x_seed_parts = x[:, :self.n_seed]
        x_seed_parts = self.cross_attn(x_seed_parts, seed_poses, seed_poses)[0] + x_seed_parts
        x = torch.cat([x_seed_parts, x[:, self.n_seed:]], dim=1)



        # time_style_embed
        x0 = x
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0
        x = self.ln1(x)

        # ff
        x0 = x
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        x = self.ln2(x)
        return x

class DiTBlock2_1(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln3 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 12)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

        self.ln_seed_poses = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_seed_poses = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.n_seed = n_seed


        self.ln_condition = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_condition = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn_condition = nn.MultiheadAttention(hidden_size, 8, batch_first=True)

    def forward(self, x, **kwargs):
        time_emb = kwargs['time_embed']
        seed_poses = kwargs['seed_poses']
        guider = kwargs['guider']

        gate1, scale1, shift1, gate2, scale2, shift2, gate3, scale3, shift3, gate4, scale4, shift4 = self.mlp(
            time_emb).chunk(12, dim=-1)
        # seed_poses
        seed_poses0 = seed_poses
        seed_poses = self.ln_seed_poses(seed_poses)
        seed_poses = seed_poses * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        seed_poses = self.ff_seed_poses(seed_poses) * gate3.unsqueeze(1) + seed_poses0
        x_seed_parts = x[:, :self.n_seed]
        x_seed_parts = self.cross_attn(x_seed_parts, seed_poses, seed_poses)[0] + x_seed_parts
        x = torch.cat([x_seed_parts, x[:, self.n_seed:]], dim=1)

        # guider
        guider0 = guider
        guider = self.ln_condition(guider)
        guider = guider * (1 + scale4.unsqueeze(1)) + shift4.unsqueeze(1)
        guider = self.ff_condition(guider) * gate4.unsqueeze(1) + guider0

        x = self.cross_attn_condition(x, guider, guider)[0] + x

        # time_style_embed
        x0 = x
        x = self.ln1(x)
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0

        # ff
        x0 = x
        x = self.ln2(x)
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        
        x = self.ln3(x)
        return x



class DiTBlock2_2(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 12)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

        self.ln_seed_poses = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_seed_poses = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.n_seed = n_seed


        self.ln_condition = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_condition = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn_condition = nn.MultiheadAttention(hidden_size, 8, batch_first=True)

    def forward(self, x, **kwargs):
        time_emb = kwargs['time_embed']
        seed_poses = kwargs['seed_poses']
        guider = kwargs['guider']

        gate1, scale1, shift1, gate2, scale2, shift2, gate3, scale3, shift3, gate4, scale4, shift4 = self.mlp(
            time_emb).chunk(12, dim=-1)
        # seed_poses
        seed_poses0 = seed_poses
        seed_poses = seed_poses * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        seed_poses = self.ff_seed_poses(seed_poses) * gate3.unsqueeze(1) + seed_poses0
        seed_poses = self.ln_seed_poses(seed_poses)
        x_seed_parts = x[:, :self.n_seed]
        x_seed_parts = self.cross_attn(x_seed_parts, seed_poses, seed_poses)[0] + x_seed_parts
        x = torch.cat([x_seed_parts, x[:, self.n_seed:]], dim=1)

        # guider
        guider0 = guider
        guider = guider * (1 + scale4.unsqueeze(1)) + shift4.unsqueeze(1)
        guider = self.ff_condition(guider) * gate4.unsqueeze(1) + guider0
        guider = self.ln_condition(guider)
        

        x = self.cross_attn_condition(x, guider, guider)[0] + x

        # time_style_embed
        x0 = x
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0
        x = self.ln1(x)
        

        # ff
        x0 = x
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        x = self.ln2(x)
        return x


class DiTBlock2_2PostNorm(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 12)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

        self.ln_seed_poses = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_seed_poses = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.n_seed = n_seed


        self.ln_condition = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_condition = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn_condition = nn.MultiheadAttention(hidden_size, 8, batch_first=True)

    def forward(self, x, **kwargs):
        time_emb = kwargs['time_embed']
        seed_poses = kwargs['seed_poses']
        guider = kwargs['guider']

        gate1, scale1, shift1, gate2, scale2, shift2, gate3, scale3, shift3, gate4, scale4, shift4 = self.mlp(
            time_emb).chunk(12, dim=-1)
        # seed_poses
        seed_poses0 = seed_poses
        seed_poses = self.ln_seed_poses(seed_poses)
        seed_poses = seed_poses * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        seed_poses = self.ff_seed_poses(seed_poses) * gate3.unsqueeze(1) + seed_poses0
        x_seed_parts = x[:, :self.n_seed]
        x_seed_parts = self.cross_attn(x_seed_parts, seed_poses, seed_poses)[0] + x_seed_parts
        x = torch.cat([x_seed_parts, x[:, self.n_seed:]], dim=1)

        # guider
        guider0 = guider
        guider = self.ln_condition(guider)
        guider = guider * (1 + scale4.unsqueeze(1)) + shift4.unsqueeze(1)
        guider = self.ff_condition(guider) * gate4.unsqueeze(1) + guider0
        
        x = self.cross_attn_condition(x, guider, guider)[0] + x

        # time_style_embed
        x0 = x
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0
        x = self.ln1(x)
        

        # ff
        x0 = x
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        x = self.ln2(x)
        return x

class DiTBlock3(nn.Module):
    def __init__(self, hidden_size=384):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln3 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 9)
        )
        self.attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

        self.ln_seed_poses = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff_seed_poses = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.cross_attn = nn.MultiheadAttention(hidden_size, 8, batch_first=True)




    def forward(self, x, **kwargs):
        time_emb = kwargs['time_embed']
        seed_poses = kwargs['seed_poses']

        gate1, scale1, shift1, gate2, scale2, shift2, gate3, scale3, shift3 = self.mlp(
            time_emb).chunk(9, dim=-1)
        # seed_poses
        seed_poses0 = seed_poses
        seed_poses = self.ln_seed_poses(seed_poses)
        seed_poses = seed_poses * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        seed_poses = self.ff_seed_poses(seed_poses) * gate3.unsqueeze(1) + seed_poses0
        x = self.cross_attn(x, seed_poses, seed_poses)[0] + x



        # time_style_embed
        x0 = x
        x = self.ln1(x)
        x = x * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.attn.forward(x, x, x)[0] * gate1.unsqueeze(1) + x0

        # ff
        x0 = x
        x = self.ln2(x)
        x = x * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = self.ff(x) * gate2.unsqueeze(1) + x0
        
        x = self.ln3(x)
        return x


   
class TimeEncoding(nn.Module):
    def __init__(self, dim, num_freqs=64):
        """
        时间编码模块，用于将连续时间 t ∈ [0,1] 映射为高维嵌入。

        参数：
        - dim: 输出嵌入的维度
        - num_freqs: 使用的频率数量（每个频率对应一个 sin 和 cos）
        """
        super(TimeEncoding, self).__init__()
        self.dim = dim
        self.num_freqs = num_freqs

        # 频率项，使用可学习的 buffer
        inv_freq = 1.0 / (10000 ** (torch.arange(0, num_freqs, dtype=torch.float32) / num_freqs))
        self.register_buffer('inv_freq', inv_freq)

        # MLP 将傅里叶特征映射到目标维度
        self.mlp = nn.Sequential(
            nn.Linear(num_freqs * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        """
        输入：
        - t: (batch_size,)，表示时间（通常归一化在 [0,1] 范围内）

        输出：
        - time_embedding: (batch_size, dim)
        """
        # 构造傅里叶特征
        x = torch.outer(t, self.inv_freq)  # (batch_size, num_freqs)
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        x = torch.cat([sin_x, cos_x], dim=-1)  # (batch_size, 2 * num_freqs)

        # 通过 MLP 映射到目标维度
        time_embedding = self.mlp(x)
        return time_embedding
    
    

class DiffusionDITNet(nn.Module):
    def __init__(self, input_channels=55 * 6 + 3 + 100, hidden_size=768, n_seed=8, num_layers=8, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_prj = nn.Linear(input_channels, hidden_size)

        self.transformer_blocks = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, input_channels)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.num_person = num_person
        self.style_embed = StyleEmbedding(num_person, 64)
        self.time_style_fuse = nn.Linear(hidden_size, hidden_size)

        self.seed_prj = nn.Linear(input_channels - 100, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.hubert_fuse = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, time_steps, **kwargs):
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(x.device)
        style_embed = self.style_embed(one_hots)
        seed_poses = self.seed_prj(seed_poses[:, :self.n_seed])

        time_style_embed = self.time_style_fuse(torch.cat([time_embed, style_embed], dim=-1))


        
        x = self.in_prj(x)
        x = self.hubert_fuse(torch.cat([x, self.hubert_prj(hubert)], dim=-1))
        # x = torch.cat([seed_poses, x], dim=1)
        x = self.pos_enc(x)
        seed_poses = self.pos_enc(seed_poses)

        kwargs = {
            "time_embed":time_style_embed,
            "seed_poses": seed_poses
        }
        for blk in self.transformer_blocks:
            x = blk(x, **kwargs)

        # x = self.head(x)[:, self.n_seed:]
        x = self.head(x)
        return x
    
class DiffusionDITNetFixedExpressions(nn.Module):
    def __init__(self, input_channels=55 * 6 + 3, hidden_size=768, n_seed=8, num_layers=8, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_prj = nn.Linear(input_channels, hidden_size)

        self.transformer_blocks = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, input_channels)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.num_person = num_person
        self.style_embed = StyleEmbedding(num_person, 64)
        self.time_style_fuse = nn.Linear(hidden_size, hidden_size)

        self.seed_prj = nn.Linear(input_channels, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.hubert_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.expressions_prj = nn.Linear(100, hidden_size)

    def forward(self, x, time_steps, **kwargs):
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']

        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(x.device)
        style_embed = self.style_embed(one_hots)
        seed_poses = self.seed_prj(seed_poses[:, :self.n_seed])

        time_style_embed = self.time_style_fuse(torch.cat([time_embed, style_embed], dim=-1))


        
        x = self.in_prj(x)
        x = self.hubert_fuse(torch.cat([x, self.hubert_prj(hubert), self.expressions_prj(expressions)], dim=-1))
        # x = torch.cat([seed_poses, x], dim=1)
        x = self.pos_enc(x)
        seed_poses = self.pos_enc(seed_poses)

        kwargs = {
            "time_embed":time_style_embed,
            "seed_poses": seed_poses
        }
        for blk in self.transformer_blocks:
            x = blk(x, **kwargs)

        # x = self.head(x)[:, self.n_seed:]
        x = self.head(x)
        return x  


class DiffusionDITNetFixedExpressions2(nn.Module):
    def __init__(self, input_channels=55 * 6 + 3, hidden_size=768, n_seed=8, num_layers=8, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_prj = nn.Linear(input_channels, hidden_size)

        self.transformer_blocks = nn.ModuleList([DiTBlock3(hidden_size=hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, input_channels)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.num_person = num_person
        self.style_embed = StyleEmbedding(num_person, 64)
        self.time_style_fuse = nn.Linear(hidden_size, hidden_size)

        self.seed_prj = nn.Linear(input_channels, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.hubert_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.expressions_prj = nn.Linear(100, hidden_size)

    def forward(self, x, time_steps, **kwargs):
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']

        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(x.device)
        style_embed = self.style_embed(one_hots)
        seed_poses = self.seed_prj(seed_poses[:, :self.n_seed])

        time_style_embed = self.time_style_fuse(torch.cat([time_embed, style_embed], dim=-1))


        
        x = self.in_prj(x)
        x = self.hubert_fuse(torch.cat([x, self.hubert_prj(hubert), self.expressions_prj(expressions)], dim=-1))
        # x = torch.cat([seed_poses, x], dim=1)
        x = self.pos_enc(x)
        seed_poses = self.pos_enc(seed_poses)

        kwargs = {
            "time_embed":time_style_embed,
            "seed_poses": seed_poses
        }
        for blk in self.transformer_blocks:
            x = blk(x, **kwargs)

        # x = self.head(x)[:, self.n_seed:]
        x = self.head(x)
        return x  
    

class DiffusionDITNetFixedExpressions3(nn.Module):
    def __init__(self, input_channels=55 * 6 + 3, hidden_size=768, n_seed=8, num_layers=8, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_prj = nn.Linear(input_channels, hidden_size)

        self.transformer_blocks = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, input_channels)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.num_person = num_person
        self.style_embed = StyleEmbedding(num_person, 64)
        self.time_style_fuse = nn.Linear(hidden_size, hidden_size)

        self.seed_prj = nn.Linear(input_channels, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.hubert_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.expressions_prj = nn.Linear(100, hidden_size)

    def forward(self, x, time_steps, **kwargs):
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']

        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(x.device)
        style_embed = self.style_embed(one_hots)
        seed_poses = self.seed_prj(seed_poses[:, :self.n_seed])

        time_style_embed = self.time_style_fuse(torch.cat([time_embed, style_embed], dim=-1))


        
        x = self.in_prj(x)
        x = self.hubert_fuse(torch.cat([x, self.hubert_prj(hubert)], dim=-1))
        # x = torch.cat([seed_poses, x], dim=1)
        x = self.pos_enc(x)
        seed_poses = self.pos_enc(seed_poses)
        expressions = self.expressions_prj(expressions)
        expressions = self.pos_enc(expressions)
        kwargs = {
            "time_embed":time_style_embed,
            "seed_poses": seed_poses,
            "guider":expressions
        }
        for blk in self.transformer_blocks:
            x = blk(x, **kwargs)

        # x = self.head(x)[:, self.n_seed:]
        x = self.head(x)
        return x  


class DiffusionDITNetFixedExpressions3_2(nn.Module):
    def __init__(self, input_channels=55 * 6 + 3, hidden_size=768, n_seed=8, num_layers=8, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_prj = nn.Linear(input_channels, hidden_size)

        self.transformer_blocks = nn.ModuleList([DiTBlock2_2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, input_channels)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.num_person = num_person
        self.style_embed = StyleEmbedding(num_person, 64)
        self.time_style_fuse = nn.Linear(hidden_size, hidden_size)

        self.seed_prj = nn.Linear(input_channels, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.hubert_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.expressions_prj = nn.Linear(100, hidden_size)

    def forward(self, x, time_steps, **kwargs):
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']

        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(x.device)
        style_embed = self.style_embed(one_hots)
        seed_poses = self.seed_prj(seed_poses[:, :self.n_seed])

        time_style_embed = self.time_style_fuse(torch.cat([time_embed, style_embed], dim=-1))


        
        x = self.in_prj(x)
        x = self.hubert_fuse(torch.cat([x, self.hubert_prj(hubert)], dim=-1))
        # x = torch.cat([seed_poses, x], dim=1)
        x = self.pos_enc(x)
        seed_poses = self.pos_enc(seed_poses)
        expressions = self.expressions_prj(expressions)
        expressions = self.pos_enc(expressions)
        kwargs = {
            "time_embed":time_style_embed,
            "seed_poses": seed_poses,
            "guider":expressions
        }
        for blk in self.transformer_blocks:
            x = blk(x, **kwargs)

        # x = self.head(x)[:, self.n_seed:]
        x = self.head(x)
        return x  



class DiffusionDITNetFixedExpressions3_3(nn.Module):
    def __init__(self, input_channels=55 * 6 + 3, hidden_size=768, n_seed=8, num_layers=8, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_prj = nn.Linear(input_channels, hidden_size)

        self.transformer_blocks = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, input_channels)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.num_person = num_person
        self.style_embed = StyleEmbedding(num_person, 64)
        self.time_style_fuse = nn.Linear(hidden_size, hidden_size)

        self.seed_prj = nn.Linear(input_channels, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.hubert_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.expressions_prj = nn.Linear(100, hidden_size)

    def forward(self, x, time_steps, **kwargs):
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']
        expressions = self.expressions_prj(expressions)
        


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(x.device)
        style_embed = self.style_embed(one_hots)
        seed_poses = self.seed_prj(seed_poses[:, :self.n_seed])

        time_style_embed = self.time_style_fuse(torch.cat([time_embed, style_embed], dim=-1))


        
        x = self.in_prj(x)
        x0 = x
        x = self.hubert_fuse(torch.cat([x, self.hubert_prj(hubert), expressions], dim=-1))
        # x = torch.cat([seed_poses, x], dim=1)
        x = self.pos_enc(x)
        expressions = self.pos_enc(expressions)
        seed_poses = self.pos_enc(seed_poses)

        kwargs = {
            "time_embed":time_style_embed,
            "seed_poses": seed_poses,
            "guider":self.pos_enc(x0)
        }
        for blk in self.transformer_blocks:
            x = blk(x, **kwargs)

        # x = self.head(x)[:, self.n_seed:]
        x = self.head(x)
        return x 

class DiffusionDITNet3(nn.Module):
    def __init__(self, input_channels=55 * 6 + 3 + 100, hidden_size=768, n_seed=8, num_layers=8, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_prj = nn.Linear(input_channels, hidden_size)

        self.transformer_blocks = nn.ModuleList([DiTBlock(hidden_size=hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, input_channels)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.num_person = num_person
        self.style_embed = StyleEmbedding(num_person, 64)
        self.time_style_fuse = nn.Linear(hidden_size, hidden_size)

        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.hubert_fuse = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, time_steps, **kwargs):
        pid = kwargs['pid']
        hubert = kwargs['hubert']
        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(x.device)
        style_embed = self.style_embed(one_hots)


        time_style_embed = self.time_style_fuse(torch.cat([time_embed, style_embed], dim=-1))

        
        x = self.in_prj(x)
        x = self.hubert_fuse(torch.cat([x, self.hubert_prj(hubert)], dim=-1))
        x = self.pos_enc(x)

        kwargs = {
            "time_embed":time_style_embed,
        }
        for blk in self.transformer_blocks:
            x = blk(x, **kwargs)
        x = self.head(x)
        return x
    
 
    
class DiffusionDITNetLatent(nn.Module):
    def __init__(self, input_channels=128, hidden_size=512, n_seed=8, num_layers=8, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_prj = nn.Linear(input_channels, hidden_size)

        self.transformer_blocks = nn.ModuleList([DiTBlock3(hidden_size=hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, input_channels)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.num_person = num_person
        self.style_embed = StyleEmbedding(num_person, 64)
        self.time_style_fuse = nn.Linear(hidden_size, hidden_size)

        self.seed_prj = nn.Linear(55 * 6 + 3, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(hidden_size, 2 *hidden_size),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size)
        )
        self.downsample_hubert = nn.Sequential(
            nn.Conv1d(1024, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
        )
        self.hubert_fuse = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, time_steps, **kwargs):
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(x.device)
        style_embed = self.style_embed(one_hots)
        seed_poses = self.seed_prj(seed_poses[:, :self.n_seed])

        time_style_embed = self.time_style_fuse(torch.cat([time_embed, style_embed], dim=-1))


        
        x = self.in_prj(x)
        hubert = self.downsample_hubert(hubert.transpose(1, 2)).transpose(1, 2)
        x = self.hubert_fuse(torch.cat([x, self.hubert_prj(hubert)], dim=-1))
        x = self.pos_enc(x)
        seed_poses = self.pos_enc(seed_poses)

        kwargs = {
            "time_embed":time_style_embed,
            "seed_poses": seed_poses
        }
        for blk in self.transformer_blocks:
            x = blk(x, **kwargs)

        # x = self.head(x)[:, self.n_seed:]
        x = self.head(x)
        return x

    

class DiffusionDITNetParts(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_upper = nn.Linear(13 * 6, hidden_size)
        self.in_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.in_face = nn.Linear(3 * 6 + 100, hidden_size)
        self.in_x_hand = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_upper = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_lower = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_face = nn.Linear(55 * 6 + 100 + 3, hidden_size)

        self.hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.upper_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.lower_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.face_fuse = nn.Linear(hidden_size * 3, hidden_size)

        self.transformer_hand = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_upper = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_lower = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_face = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.upper = nn.Linear(hidden_size, 13 * 6)
        self.lower = nn.Linear(hidden_size, 9 * 6 + 3)
        self.face = nn.Linear(hidden_size, 3 * 6 + 100)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_upper = nn.Linear(13 * 6, hidden_size)
        self.seed_prj_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.seed_prj_face = nn.Linear(3 * 6, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        
        x0 = x
        hand = x[..., hand_indices]
        upper = x[..., upper_indices]
        lower = x[..., lower_indices]
        face = x[..., face_indices]

        hand = self.in_hand(hand)
        upper = self.in_upper(upper)
        lower = self.in_lower(lower)
        face = self.in_face(face)
        
        xhand = self.in_x_hand(x) 
        xupper = self.in_x_upper(x)
        xlower = self.in_x_lower(x)
        xface = self.in_x_face(x)


        huberts = self.hubert_prj(hubert)

        hand = self.hand_fuse(torch.cat([hand, xhand, huberts], dim=-1))
        upper = self.upper_fuse(torch.cat([upper, xupper, huberts], dim=-1))
        lower = self.lower_fuse(torch.cat([lower, xlower, huberts], dim=-1))
        face = self.face_fuse(torch.cat([face, xface, huberts], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_hand_poses = self.seed_prj_hand(seed_poses[:, :self.n_seed, hand_indices])
        seed_upper_poses = self.seed_prj_upper(seed_poses[:, :self.n_seed, upper_indices])
        seed_lower_poses = self.seed_prj_lower(seed_poses[:, :self.n_seed, lower_indices])
        seed_face_poses = self.seed_prj_face(seed_poses[:, :self.n_seed, face_joint_indices])

        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_upper_poses = self.pos_enc(seed_upper_poses)
        seed_lower_poses = self.pos_enc(seed_lower_poses)
        seed_face_poses = self.pos_enc(seed_face_poses)

        hand = self.pos_enc(hand)
        upper = self.pos_enc(upper)
        lower = self.pos_enc(lower)
        face = self.pos_enc(face)

        

        kwargs = {'seed_poses': seed_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_hand:
            hand = blk(hand, **kwargs)
        kwargs = {'seed_poses': seed_upper_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_upper:
            upper = blk(upper, **kwargs)
        kwargs = {'seed_poses': seed_lower_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_lower:
            lower = blk(lower, **kwargs)
        kwargs = {'seed_poses': seed_face_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_face:
            face = blk(face, **kwargs)

        hand = self.hand(hand)
        upper = self.upper(upper)
        lower = self.lower(lower)
        face = self.face(face)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., upper_indices] = upper
        x[..., lower_indices] = lower
        x[..., face_indices] = face
        return x

class DiffusionDITNetPartsFixedExpressions(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_upper = nn.Linear(13 * 6, hidden_size)
        self.in_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.in_face = nn.Linear(3 * 6 + 100, hidden_size)
        
        self.in_x_hand = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_upper = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_lower = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_face = nn.Linear(55 * 6 + 100 + 3, hidden_size)

        self.hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.upper_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.lower_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.face_fuse = nn.Linear(hidden_size * 3, hidden_size)

        self.transformer_hand = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_upper = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_lower = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_face = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.upper = nn.Linear(hidden_size, 13 * 6)
        self.lower = nn.Linear(hidden_size, 9 * 6 + 3)
        self.face = nn.Linear(hidden_size, 3 * 6)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_upper = nn.Linear(13 * 6, hidden_size)
        self.seed_prj_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.seed_prj_face = nn.Linear(3 * 6, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']
        
        x0 = x
        hand = x[..., hand_indices]
        upper = x[..., upper_indices]
        lower = x[..., lower_indices]
        face = torch.cat([x[..., face_joint_indices], expressions], dim=-1)

        hand = self.in_hand(hand)
        upper = self.in_upper(upper)
        lower = self.in_lower(lower)
        face = self.in_face(face)
        
        x = torch.cat([x, expressions], dim=-1)
        xhand = self.in_x_hand(x) 
        xupper = self.in_x_upper(x)
        xlower = self.in_x_lower(x)
        xface = self.in_x_face(x)


        huberts = self.hubert_prj(hubert)

        hand = self.hand_fuse(torch.cat([hand, xhand, huberts], dim=-1))
        upper = self.upper_fuse(torch.cat([upper, xupper, huberts], dim=-1))
        lower = self.lower_fuse(torch.cat([lower, xlower, huberts], dim=-1))
        face = self.face_fuse(torch.cat([face, xface, huberts], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_hand_poses = self.seed_prj_hand(seed_poses[:, :self.n_seed, hand_indices])
        seed_upper_poses = self.seed_prj_upper(seed_poses[:, :self.n_seed, upper_indices])
        seed_lower_poses = self.seed_prj_lower(seed_poses[:, :self.n_seed, lower_indices])
        seed_face_poses = self.seed_prj_face(seed_poses[:, :self.n_seed, face_joint_indices])

        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_upper_poses = self.pos_enc(seed_upper_poses)
        seed_lower_poses = self.pos_enc(seed_lower_poses)
        seed_face_poses = self.pos_enc(seed_face_poses)

        hand = self.pos_enc(hand)
        upper = self.pos_enc(upper)
        lower = self.pos_enc(lower)
        face = self.pos_enc(face)

        

        kwargs = {'seed_poses': seed_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_hand:
            hand = blk(hand, **kwargs)
        kwargs = {'seed_poses': seed_upper_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_upper:
            upper = blk(upper, **kwargs)
        kwargs = {'seed_poses': seed_lower_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_lower:
            lower = blk(lower, **kwargs)
        kwargs = {'seed_poses': seed_face_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_face:
            face = blk(face, **kwargs)

        hand = self.hand(hand)
        upper = self.upper(upper)
        lower = self.lower(lower)
        face = self.face(face)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., upper_indices] = upper
        x[..., lower_indices] = lower
        x[..., face_joint_indices] = face
        return x


class DiffusionDITNetPartsFixedExpressions2InFilling(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)
        
        self.in_x_hand = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_body = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_mask = nn.Linear(55 * 6 + 3, hidden_size)
        self.masked_val = nn.Parameter(torch.randn(1,1,hidden_size))


        self.hand_fuse = nn.Linear(hidden_size * 4, hidden_size)
        self.body_fuse = nn.Linear(hidden_size * 4, hidden_size)


        self.transformer_hand = nn.ModuleList([DiTBlock(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_body = nn.ModuleList([DiTBlock(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.body = nn.Linear(hidden_size, (13 + 9 + 3) * 6 + 3)


        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person



        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        # seed_poses = kwargs['seed_poses']
        mask = kwargs['mask'] # B, L
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']
        x1 = kwargs['x1']
        body_indices = np.concatenate([upper_indices, lower_indices, face_joint_indices], axis=0)
        x0 = x

        hand = x[..., hand_indices]
        body = x[..., body_indices]

        hand = self.in_hand(hand)
        body = self.in_body(body)
        
        x = torch.cat([x, expressions], dim=-1)
        xhand = self.in_x_hand(x) 
        xbody = self.in_x_body(x)

        masked_x = torch.where(mask.unsqueeze(-1), self.masked_val, self.in_x_mask(x1))

        huberts = self.hubert_prj(hubert)
        hand = self.hand_fuse(torch.cat([hand, xhand, huberts, masked_x], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, huberts, masked_x], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)


        hand = self.pos_enc(hand)
        body = self.pos_enc(body)

        kwargs = {'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_hand:
            hand = blk(hand, **kwargs)

        kwargs = {'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_body:
            body = blk(body, **kwargs)


        hand = self.hand(hand)
        body = self.body(body)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., body_indices] = body
        return x

class DiffusionDITNetPartsFixedExpressions2(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)
        
        self.in_x_hand = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_body = nn.Linear(55 * 6 + 100 + 3, hidden_size)


        self.hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.body_fuse = nn.Linear(hidden_size * 3, hidden_size)


        self.transformer_hand = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_body = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.body = nn.Linear(hidden_size, (13 + 9 + 3) * 6 + 3)


        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)

        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']
        body_indices = np.concatenate([upper_indices, lower_indices, face_joint_indices], axis=0)
        x0 = x

        hand = x[..., hand_indices]
        body = x[..., body_indices]

        hand = self.in_hand(hand)
        body = self.in_body(body)
        
        x = torch.cat([x, expressions], dim=-1)
        xhand = self.in_x_hand(x) 
        xbody = self.in_x_body(x)



        huberts = self.hubert_prj(hubert)
        hand = self.hand_fuse(torch.cat([hand, xhand, huberts], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, huberts], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_hand_poses = self.seed_prj_hand(seed_poses[:, :self.n_seed, hand_indices])
        seed_body_poses = self.seed_prj_body(seed_poses[:, :self.n_seed, body_indices])


        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_body_poses = self.pos_enc(seed_body_poses)


        hand = self.pos_enc(hand)
        body = self.pos_enc(body)


        

        kwargs = {'seed_poses': seed_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_hand:
            hand = blk(hand, **kwargs)

        kwargs = {'seed_poses': seed_body_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_body:
            body = blk(body, **kwargs)


        hand = self.hand(hand)
        body = self.body(body)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., body_indices] = body
        return x


class DiffusionDITNetPartsFixedExpressions2PostNorm(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)
        
        self.in_x_hand = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_body = nn.Linear(55 * 6 + 100 + 3, hidden_size)


        self.hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.body_fuse = nn.Linear(hidden_size * 3, hidden_size)


        self.transformer_hand = nn.ModuleList([DiTBlock2PostNorm(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_body = nn.ModuleList([DiTBlock2PostNorm(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.body = nn.Linear(hidden_size, (13 + 9 + 3) * 6 + 3)


        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)

        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']
        body_indices = np.concatenate([upper_indices, lower_indices, face_joint_indices], axis=0)
        x0 = x

        hand = x[..., hand_indices]
        body = x[..., body_indices]

        hand = self.in_hand(hand)
        body = self.in_body(body)
        
        x = torch.cat([x, expressions], dim=-1)
        xhand = self.in_x_hand(x) 
        xbody = self.in_x_body(x)



        huberts = self.hubert_prj(hubert)
        hand = self.hand_fuse(torch.cat([hand, xhand, huberts], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, huberts], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_hand_poses = self.seed_prj_hand(seed_poses[:, :self.n_seed, hand_indices])
        seed_body_poses = self.seed_prj_body(seed_poses[:, :self.n_seed, body_indices])


        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_body_poses = self.pos_enc(seed_body_poses)


        hand = self.pos_enc(hand)
        body = self.pos_enc(body)


        

        kwargs = {'seed_poses': seed_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_hand:
            hand = blk(hand, **kwargs)

        kwargs = {'seed_poses': seed_body_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_body:
            body = blk(body, **kwargs)


        hand = self.hand(hand)
        body = self.body(body)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., body_indices] = body
        return x

class DiffusionDITNetPartsFixedExpressions3(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)
        
        self.in_x_hand = nn.Linear(55 * 6  + 3, hidden_size)
        self.in_x_body = nn.Linear(55 * 6  + 3, hidden_size)

        self.in_expression_hand = nn.Linear(100, hidden_size)
        self.in_expression_body = nn.Linear(100, hidden_size)

        self.hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.body_fuse = nn.Linear(hidden_size * 3, hidden_size)


        self.transformer_hand = nn.ModuleList([DiTBlock2_2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_body = nn.ModuleList([DiTBlock2_2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.body = nn.Linear(hidden_size, (13 + 9 + 3) * 6 + 3)


        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)

        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']
        body_indices = np.concatenate([upper_indices, lower_indices, face_joint_indices], axis=0)
        x0 = x

        hand = x[..., hand_indices]
        body = x[..., body_indices]

        hand = self.in_hand(hand)
        body = self.in_body(body)
        

        xhand = self.in_x_hand(x) 
        xbody = self.in_x_body(x)



        huberts = self.hubert_prj(hubert)
        hand = self.hand_fuse(torch.cat([hand, xhand, huberts], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, huberts], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_hand_poses = self.seed_prj_hand(seed_poses[:, :self.n_seed, hand_indices])
        seed_body_poses = self.seed_prj_body(seed_poses[:, :self.n_seed, body_indices])


        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_body_poses = self.pos_enc(seed_body_poses)


        hand = self.pos_enc(hand)
        body = self.pos_enc(body)

        hand_expressions = self.pos_enc(self.in_expression_hand(expressions))
        body_expressions = self.pos_enc(self.in_expression_body(expressions))

        

        kwargs = {'seed_poses': seed_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider":hand_expressions}
        for blk in self.transformer_hand:
            hand = blk(hand, **kwargs)

        kwargs = {'seed_poses': seed_body_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider":body_expressions}
        for blk in self.transformer_body:
            body = blk(body, **kwargs)


        hand = self.hand(hand)
        body = self.body(body)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., body_indices] = body
        return x


class DiffusionDITNetPartsFixedExpressions3PreNorm(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)
        
        self.in_x_hand = nn.Linear(55 * 6  + 3, hidden_size)
        self.in_x_body = nn.Linear(55 * 6  + 3, hidden_size)

        self.in_expression_hand = nn.Linear(100, hidden_size)
        self.in_expression_body = nn.Linear(100, hidden_size)

        self.hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.body_fuse = nn.Linear(hidden_size * 3, hidden_size)


        self.transformer_hand = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_body = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.body = nn.Linear(hidden_size, (13 + 9 + 3) * 6 + 3)


        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)

        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']
        body_indices = np.concatenate([upper_indices, lower_indices, face_joint_indices], axis=0)
        x0 = x

        hand = x[..., hand_indices]
        body = x[..., body_indices]

        hand = self.in_hand(hand)
        body = self.in_body(body)
        

        xhand = self.in_x_hand(x) 
        xbody = self.in_x_body(x)



        huberts = self.hubert_prj(hubert)
        hand = self.hand_fuse(torch.cat([hand, xhand, huberts], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, huberts], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_hand_poses = self.seed_prj_hand(seed_poses[:, :self.n_seed, hand_indices])
        seed_body_poses = self.seed_prj_body(seed_poses[:, :self.n_seed, body_indices])


        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_body_poses = self.pos_enc(seed_body_poses)


        hand = self.pos_enc(hand)
        body = self.pos_enc(body)

        hand_expressions = self.pos_enc(self.in_expression_hand(expressions))
        body_expressions = self.pos_enc(self.in_expression_body(expressions))

        

        kwargs = {'seed_poses': seed_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider":hand_expressions}
        for blk in self.transformer_hand:
            hand = blk(hand, **kwargs)

        kwargs = {'seed_poses': seed_body_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider":body_expressions}
        for blk in self.transformer_body:
            body = blk(body, **kwargs)


        hand = self.hand(hand)
        body = self.body(body)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., body_indices] = body
        return x


class DiffusionDITNetPartsFixedExpressions3PostNorm(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)
        
        self.in_x_hand = nn.Linear(55 * 6  + 3, hidden_size)
        self.in_x_body = nn.Linear(55 * 6  + 3, hidden_size)

        self.in_expression_hand = nn.Linear(100, hidden_size)
        self.in_expression_body = nn.Linear(100, hidden_size)

        self.hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.body_fuse = nn.Linear(hidden_size * 3, hidden_size)


        self.transformer_hand = nn.ModuleList([DiTBlock2_2PostNorm(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_body = nn.ModuleList([DiTBlock2_2PostNorm(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.body = nn.Linear(hidden_size, (13 + 9 + 3) * 6 + 3)


        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)

        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']
        body_indices = np.concatenate([upper_indices, lower_indices, face_joint_indices], axis=0)
        x0 = x

        hand = x[..., hand_indices]
        body = x[..., body_indices]

        hand = self.in_hand(hand)
        body = self.in_body(body)
        

        xhand = self.in_x_hand(x) 
        xbody = self.in_x_body(x)



        huberts = self.hubert_prj(hubert)
        hand = self.hand_fuse(torch.cat([hand, xhand, huberts], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, huberts], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_hand_poses = self.seed_prj_hand(seed_poses[:, :self.n_seed, hand_indices])
        seed_body_poses = self.seed_prj_body(seed_poses[:, :self.n_seed, body_indices])


        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_body_poses = self.pos_enc(seed_body_poses)


        hand = self.pos_enc(hand)
        body = self.pos_enc(body)

        hand_expressions = self.pos_enc(self.in_expression_hand(expressions))
        body_expressions = self.pos_enc(self.in_expression_body(expressions))

        

        kwargs = {'seed_poses': seed_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider":hand_expressions}
        for blk in self.transformer_hand:
            hand = blk(hand, **kwargs)

        kwargs = {'seed_poses': seed_body_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider":body_expressions}
        for blk in self.transformer_body:
            body = blk(body, **kwargs)


        hand = self.hand(hand)
        body = self.body(body)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., body_indices] = body
        return x


class DiffusionDITNetParts5FixedExpressions(nn.Module):
    def __init__(self, hidden_size=256, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_left_hand = nn.Linear(15 * 6, hidden_size)
        self.in_right_hand = nn.Linear(15 * 6, hidden_size)
        self.in_upper = nn.Linear(13 * 6, hidden_size)
        self.in_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.in_face = nn.Linear(3 * 6 + 100, hidden_size)
        
        self.in_x_left_hand = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_right_hand = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_upper = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_lower = nn.Linear(55 * 6 + 100 + 3, hidden_size)
        self.in_x_face = nn.Linear(55 * 6 + 100 + 3, hidden_size)

        self.left_hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.right_hand_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.upper_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.lower_fuse = nn.Linear(hidden_size * 3, hidden_size)
        self.face_fuse = nn.Linear(hidden_size * 3, hidden_size)

        self.transformer_left_hand = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_right_hand = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_upper = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_lower = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_face = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.left_hand = nn.Linear(hidden_size, 15 * 6)
        self.right_hand = nn.Linear(hidden_size, 15 * 6)
        self.upper = nn.Linear(hidden_size, 13 * 6)
        self.lower = nn.Linear(hidden_size, 9 * 6 + 3)
        self.face = nn.Linear(hidden_size, 3 * 6)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_left_hand = nn.Linear(15 * 6, hidden_size)
        self.seed_prj_right_hand = nn.Linear(15 * 6, hidden_size)
        self.seed_prj_upper = nn.Linear(13 * 6, hidden_size)
        self.seed_prj_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.seed_prj_face = nn.Linear(3 * 6, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        # hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        # face_indices = kwargs['face_indices']
        lhand_indices = kwargs['lhand_indices']
        rhand_indices = kwargs['rhand_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expressions = kwargs['expressions']
        
        x0 = x
        left_hand = x[..., lhand_indices]
        right_hand = x[..., rhand_indices]
        upper = x[..., upper_indices]
        lower = x[..., lower_indices]
        face = torch.cat([x[..., face_joint_indices], expressions], dim=-1)

        left_hand = self.in_left_hand(left_hand)
        right_hand = self.in_right_hand(right_hand)
        upper = self.in_upper(upper)
        lower = self.in_lower(lower)
        face = self.in_face(face)
        
        x = torch.cat([x, expressions], dim=-1)
        xleft_hand = self.in_x_left_hand(x) 
        xright_hand = self.in_x_right_hand(x) 
        xupper = self.in_x_upper(x)
        xlower = self.in_x_lower(x)
        xface = self.in_x_face(x)


        huberts = self.hubert_prj(hubert)

        left_hand = self.left_hand_fuse(torch.cat([left_hand, xleft_hand, huberts], dim=-1))
        right_hand = self.right_hand_fuse(torch.cat([right_hand, xright_hand, huberts], dim=-1))
        upper = self.upper_fuse(torch.cat([upper, xupper, huberts], dim=-1))
        lower = self.lower_fuse(torch.cat([lower, xlower, huberts], dim=-1))
        face = self.face_fuse(torch.cat([face, xface, huberts], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_left_hand_poses = self.seed_prj_left_hand(seed_poses[:, :self.n_seed, lhand_indices])
        seed_right_hand_poses = self.seed_prj_right_hand(seed_poses[:, :self.n_seed, rhand_indices])
        seed_upper_poses = self.seed_prj_upper(seed_poses[:, :self.n_seed, upper_indices])
        seed_lower_poses = self.seed_prj_lower(seed_poses[:, :self.n_seed, lower_indices])
        seed_face_poses = self.seed_prj_face(seed_poses[:, :self.n_seed, face_joint_indices])

        seed_left_hand_poses = self.pos_enc(seed_left_hand_poses)
        seed_right_hand_poses = self.pos_enc(seed_right_hand_poses)
        seed_upper_poses = self.pos_enc(seed_upper_poses)
        seed_lower_poses = self.pos_enc(seed_lower_poses)
        seed_face_poses = self.pos_enc(seed_face_poses)

        left_hand = self.pos_enc(left_hand)
        right_hand = self.pos_enc(right_hand)
        upper = self.pos_enc(upper)
        lower = self.pos_enc(lower)
        face = self.pos_enc(face)

        

        kwargs = {'seed_poses': seed_left_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_left_hand:
            left_hand = blk(left_hand, **kwargs)
        kwargs = {'seed_poses': seed_right_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_right_hand:
            right_hand = blk(right_hand, **kwargs)
        kwargs = {'seed_poses': seed_upper_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_upper:
            upper = blk(upper, **kwargs)
        kwargs = {'seed_poses': seed_lower_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_lower:
            lower = blk(lower, **kwargs)
        kwargs = {'seed_poses': seed_face_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
        for blk in self.transformer_face:
            face = blk(face, **kwargs)

        left_hand = self.left_hand(left_hand)
        right_hand = self.right_hand(right_hand)
        upper = self.upper(upper)
        lower = self.lower(lower)
        face = self.face(face)

        x = torch.zeros(x0.shape, dtype=face.dtype, device=face.device)
        x[..., lhand_indices] = left_hand
        x[..., rhand_indices] = right_hand
        x[..., upper_indices] = upper
        x[..., lower_indices] = lower
        x[..., face_joint_indices] = face
        return x

class DiffusionDITNetPartsHierarchy(nn.Module):
    def __init__(self, hidden_size=384, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_upper = nn.Linear(13 * 6, hidden_size)
        self.in_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.in_face = nn.Linear(3 * 6 + 100, hidden_size)


        self.hand_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.upper_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.lower_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.face_fuse = nn.Linear(hidden_size * 2, hidden_size)

        self.transformer_hand = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_upper = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_lower = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_face = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.upper = nn.Linear(hidden_size, 13 * 6)
        self.lower = nn.Linear(hidden_size, 9 * 6 + 3)
        self.face = nn.Linear(hidden_size, 3 * 6 + 100)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_upper = nn.Linear(13 * 6, hidden_size)
        self.seed_prj_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.seed_prj_face = nn.Linear(3 * 6, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        
        x0 = x
        hand = x[..., hand_indices]
        upper = x[..., upper_indices]
        lower = x[..., lower_indices]
        face = x[..., face_indices]

        hand = self.in_hand(hand)
        upper = self.in_upper(upper)
        lower = self.in_lower(lower)
        face = self.in_face(face)



        huberts = self.hubert_prj(hubert)

        hand = self.hand_fuse(torch.cat([hand,  huberts], dim=-1))
        upper = self.upper_fuse(torch.cat([upper,  huberts], dim=-1))
        lower = self.lower_fuse(torch.cat([lower,  huberts], dim=-1))
        face = self.face_fuse(torch.cat([face, huberts], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_hand_poses = self.seed_prj_hand(seed_poses[:, :self.n_seed, hand_indices])
        seed_upper_poses = self.seed_prj_upper(seed_poses[:, :self.n_seed, upper_indices])
        seed_lower_poses = self.seed_prj_lower(seed_poses[:, :self.n_seed, lower_indices])
        seed_face_poses = self.seed_prj_face(seed_poses[:, :self.n_seed, face_joint_indices])

        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_upper_poses = self.pos_enc(seed_upper_poses)
        seed_lower_poses = self.pos_enc(seed_lower_poses)
        seed_face_poses = self.pos_enc(seed_face_poses)

        hand = self.pos_enc(hand)
        upper = self.pos_enc(upper)
        lower = self.pos_enc(lower)
        face = self.pos_enc(face)

        
        for face_blk, hand_blk, upper_blk, lower_blk in zip(self.transformer_face, self.transformer_hand, self.transformer_upper, self.transformer_lower):
            kwargs = {'seed_poses': seed_face_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts}
            face = face_blk(face, **kwargs)
            kwargs = {'seed_poses': seed_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": face}
            hand = hand_blk(hand, **kwargs)
            kwargs = {'seed_poses': seed_upper_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": hand}
            upper = upper_blk(upper, **kwargs)
            kwargs = {'seed_poses': seed_lower_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": upper}
            lower = lower_blk(lower, **kwargs)




        hand = self.hand(hand)
        upper = self.upper(upper)
        lower = self.lower(lower)
        face = self.face(face)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., upper_indices] = upper
        x[..., lower_indices] = lower
        x[..., face_indices] = face
        return x



class DiffusionDITNetPartsHierarchy2(nn.Module):
    def __init__(self, hidden_size=256, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_upper = nn.Linear(13 * 6, hidden_size)
        self.in_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.in_face = nn.Linear(3 * 6, hidden_size)


        self.hand_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.upper_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.lower_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.face_fuse = nn.Linear(hidden_size * 2, hidden_size)


        self.face_guider_prj = nn.ModuleList([nn.Linear(100, hidden_size) for _ in range(num_layers)])
        self.hand_guider_prj = nn.ModuleList([nn.Linear(100 + hidden_size, hidden_size) for _ in range(num_layers)])
        self.upper_guider_prj = nn.ModuleList([nn.Linear(100 + hidden_size*2, hidden_size) for _ in range(num_layers)])
        self.lower_guider_prj = nn.ModuleList([nn.Linear(100 + hidden_size*3, hidden_size) for _ in range(num_layers)])


        self.transformer_hand = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_upper = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_lower = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_face = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.upper = nn.Linear(hidden_size, 13 * 6)
        self.lower = nn.Linear(hidden_size, 9 * 6 + 3)
        self.face = nn.Linear(hidden_size, 3 * 6)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_upper = nn.Linear(13 * 6, hidden_size)
        self.seed_prj_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.seed_prj_face = nn.Linear(3 * 6, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expression = kwargs['expressions']
        
        x0 = x
        hand = x[..., hand_indices]
        upper = x[..., upper_indices]
        lower = x[..., lower_indices]
        face = x[..., face_joint_indices]

        hand = self.in_hand(hand)
        upper = self.in_upper(upper)
        lower = self.in_lower(lower)
        face = self.in_face(face)




        huberts = self.hubert_prj(hubert)

        hand = self.hand_fuse(torch.cat([hand,  huberts], dim=-1))
        upper = self.upper_fuse(torch.cat([upper,  huberts], dim=-1))
        lower = self.lower_fuse(torch.cat([lower,  huberts], dim=-1))
        face = self.face_fuse(torch.cat([face, huberts], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_hand_poses = self.seed_prj_hand(seed_poses[:, :self.n_seed, hand_indices])
        seed_upper_poses = self.seed_prj_upper(seed_poses[:, :self.n_seed, upper_indices])
        seed_lower_poses = self.seed_prj_lower(seed_poses[:, :self.n_seed, lower_indices])
        seed_face_poses = self.seed_prj_face(seed_poses[:, :self.n_seed, face_joint_indices])

        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_upper_poses = self.pos_enc(seed_upper_poses)
        seed_lower_poses = self.pos_enc(seed_lower_poses)
        seed_face_poses = self.pos_enc(seed_face_poses)

        hand = self.pos_enc(hand)
        upper = self.pos_enc(upper)
        lower = self.pos_enc(lower)
        face = self.pos_enc(face)

        
        blk_id = 0
        for face_blk, hand_blk, upper_blk, lower_blk in zip(self.transformer_face, self.transformer_hand, self.transformer_upper, self.transformer_lower):
            face_guider = self.face_guider_prj[blk_id](expression)
            kwargs = {'seed_poses': seed_face_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": face_guider}
            face = face_blk(face, **kwargs)

            hand_guider = self.hand_guider_prj[blk_id](torch.cat([face, expression], dim=-1))
            kwargs = {'seed_poses': seed_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": hand_guider}
            hand = hand_blk(hand, **kwargs)

            upper_guider = self.upper_guider_prj[blk_id](torch.cat([face, hand, expression], dim=-1))
            kwargs = {'seed_poses': seed_upper_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": upper_guider}
            upper = upper_blk(upper, **kwargs)

            lower_guider = self.lower_guider_prj[blk_id](torch.cat([face, hand, upper, expression], dim=-1))
            kwargs = {'seed_poses': seed_lower_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": lower_guider}
            lower = lower_blk(lower, **kwargs)

            blk_id +=1




        hand = self.hand(hand)
        upper = self.upper(upper)
        lower = self.lower(lower)
        face = self.face(face)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., upper_indices] = upper
        x[..., lower_indices] = lower
        x[..., face_joint_indices] = face
        return x 
    


class DiffusionDITNetPartsHierarchy3(nn.Module):
    def __init__(self, hidden_size=256, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_upper = nn.Linear(13 * 6, hidden_size)
        self.in_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.in_face = nn.Linear(3 * 6, hidden_size)


        self.hand_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.upper_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.lower_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.face_fuse = nn.Linear(hidden_size * 2, hidden_size)


        self.face_guider_prj = nn.ModuleList([nn.Linear(100, hidden_size) for _ in range(num_layers)])
        self.hand_guider_prj = nn.ModuleList([nn.Linear(100 + hidden_size, hidden_size) for _ in range(num_layers)])
        self.upper_guider_prj = nn.ModuleList([nn.Linear(100 + hidden_size*2, hidden_size) for _ in range(num_layers)])
        self.lower_guider_prj = nn.ModuleList([nn.Linear(100 + hidden_size*3, hidden_size) for _ in range(num_layers)])


        self.transformer_hand = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_upper = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_lower = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_face = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.upper = nn.Linear(hidden_size, 13 * 6)
        self.lower = nn.Linear(hidden_size, 9 * 6 + 3)
        self.face = nn.Linear(hidden_size, 3 * 6)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_upper = nn.Linear(13 * 6, hidden_size)
        self.seed_prj_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.seed_prj_face = nn.Linear(3 * 6, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expression = kwargs['expressions']
        
        x0 = x
        hand = x[..., hand_indices]
        upper = x[..., upper_indices]
        lower = x[..., lower_indices]
        face = x[..., face_joint_indices]

        hand = self.in_hand(hand)
        upper = self.in_upper(upper)
        lower = self.in_lower(lower)
        face = self.in_face(face)




        huberts = self.hubert_prj(hubert)

        hand = self.hand_fuse(torch.cat([hand,  huberts], dim=-1))
        upper = self.upper_fuse(torch.cat([upper,  huberts], dim=-1))
        lower = self.lower_fuse(torch.cat([lower,  huberts], dim=-1))
        face = self.face_fuse(torch.cat([face, huberts], dim=-1))


        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_hand_poses = self.seed_prj_hand(seed_poses[:, :self.n_seed, hand_indices])
        seed_upper_poses = self.seed_prj_upper(seed_poses[:, :self.n_seed, upper_indices])
        seed_lower_poses = self.seed_prj_lower(seed_poses[:, :self.n_seed, lower_indices])
        seed_face_poses = self.seed_prj_face(seed_poses[:, :self.n_seed, face_joint_indices])

        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_upper_poses = self.pos_enc(seed_upper_poses)
        seed_lower_poses = self.pos_enc(seed_lower_poses)
        seed_face_poses = self.pos_enc(seed_face_poses)

        hand = self.pos_enc(hand)
        upper = self.pos_enc(upper)
        lower = self.pos_enc(lower)
        face = self.pos_enc(face)

        
        blk_id = 0
        for face_blk, hand_blk, upper_blk, lower_blk in zip(self.transformer_face, self.transformer_hand, self.transformer_upper, self.transformer_lower):
            face_guider = self.face_guider_prj[blk_id](expression)
            face_guider = self.pos_enc(face_guider)
            kwargs = {'seed_poses': seed_face_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": face_guider}
            face = face_blk(face, **kwargs)

            hand_guider = self.hand_guider_prj[blk_id](torch.cat([face, expression], dim=-1))
            hand_guider = self.pos_enc(hand_guider)
            kwargs = {'seed_poses': seed_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": hand_guider}
            hand = hand_blk(hand, **kwargs)

            upper_guider = self.upper_guider_prj[blk_id](torch.cat([face, hand, expression], dim=-1))
            upper_guider = self.pos_enc(upper_guider)
            kwargs = {'seed_poses': seed_upper_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": upper_guider}
            upper = upper_blk(upper, **kwargs)

            lower_guider = self.lower_guider_prj[blk_id](torch.cat([face, hand, upper, expression], dim=-1))
            lower_guider = self.pos_enc(lower_guider)
            kwargs = {'seed_poses': seed_lower_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": lower_guider}
            lower = lower_blk(lower, **kwargs)

            blk_id +=1




        hand = self.hand(hand)
        upper = self.upper(upper)
        lower = self.lower(lower)
        face = self.face(face)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., upper_indices] = upper
        x[..., lower_indices] = lower
        x[..., face_joint_indices] = face
        return x 




class DiffusionDITNetPartsHierarchy4(nn.Module):
    def __init__(self, hidden_size=256, n_seed=8, num_layers=6, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)

        self.hand_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.body_fuse = nn.Linear(hidden_size * 2, hidden_size)


        self.body_guider_prj = nn.ModuleList([nn.Linear(100, hidden_size) for _ in range(num_layers)])
        self.hand_guider_prj = nn.ModuleList([nn.Linear(100 + hidden_size, hidden_size) for _ in range(num_layers)])



        self.transformer_hand = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])
        self.transformer_body = nn.ModuleList([DiTBlock2_1(hidden_size=hidden_size) for _ in range(num_layers)])

        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.body = nn.Linear(hidden_size, (13 + 9 + 3) * 6 + 3)


        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_body = nn.Linear((13 + 9 + 3) * 6 + 3, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']

        body_indices = np.concatenate([upper_indices, lower_indices, face_joint_indices], axis=-1)
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        expression = kwargs['expressions']
        
        x0 = x
        hand = x[..., hand_indices]
        body = x[..., body_indices]


        hand = self.in_hand(hand)
        body = self.in_body(body)


        huberts = self.hubert_prj(hubert)

        hand = self.hand_fuse(torch.cat([hand,  huberts], dim=-1))
        body = self.body_fuse(torch.cat([body,  huberts], dim=-1))



        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        time_style_embed = torch.cat([time_embed, self.style_embed(one_hots)], dim=-1)

        seed_hand_poses = self.seed_prj_hand(seed_poses[:, :self.n_seed, hand_indices])
        seed_body_poses = self.seed_prj_body(seed_poses[:, :self.n_seed, body_indices])


        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_body_poses = self.pos_enc(seed_body_poses)


        hand = self.pos_enc(hand)
        body = self.pos_enc(body)


        
        blk_id = 0
        for hand_blk, body_blk,in zip(self.transformer_hand, self.transformer_body):
            body_guider = self.body_guider_prj[blk_id](expression)
            body_guider = self.pos_enc(body_guider)
            kwargs = {'seed_poses': seed_body_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": body_guider}
            body = body_blk(body, **kwargs)

            hand_guider = self.hand_guider_prj[blk_id](torch.cat([expression, body.detach()], dim=-1))
            hand_guider = self.pos_enc(hand_guider)
            kwargs = {'seed_poses': seed_hand_poses, 'time_embed': time_style_embed, 'audio_embeding': huberts, "guider": hand_guider}
            hand = hand_blk(hand, **kwargs)

            blk_id +=1




        hand = self.hand(hand)
        body = self.body(body)

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., body_indices] = body
        return x 


class DiffusionDITNet2(nn.Module):
    def __init__(self, input_channels=55 * 6 + 3 + 100, hidden_size=768, n_seed=8, num_layers=8):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_prj = nn.Linear(input_channels, hidden_size)

        self.transformer_blocks = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, input_channels)

        self.time_embed = TimeEncoding(hidden_size - 64, num_freqs=128)

        self.style_embed = StyleEmbedding(31, 64)
        self.time_style_fuse = nn.Linear(hidden_size, hidden_size)

        self.seed_prj = nn.Linear(input_channels - 100, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.hubert_fuse = nn.Linear(hidden_size * 2 + 256, hidden_size)

    def forward(self, x, time_steps, **kwargs):
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        clip_feat = kwargs['clip']
        time_embed = self.time_embed(time_steps)
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        style_embed = self.style_embed(one_hots)
        seed_poses = self.seed_prj(seed_poses[:, :self.n_seed])

        time_style_embed = self.time_style_fuse(torch.cat([time_embed, style_embed], dim=-1))


        
        x = self.in_prj(x)
        clip_feat = clip_feat.reshape((-1,1, 256)).expand(-1, x.shape[1], -1)
        x = self.hubert_fuse(torch.cat([x, self.hubert_prj(hubert), clip_feat], dim=-1))
        # x = torch.cat([seed_poses, x], dim=1)
        x = self.pos_enc(x)
        seed_poses = self.pos_enc(seed_poses)

        kwargs = {
            "time_embed":time_style_embed,
            "seed_poses": seed_poses
        }
        for blk in self.transformer_blocks:
            x = blk(x, **kwargs)

        # x = self.head(x)[:, self.n_seed:]
        x = self.head(x)
        return x
    
class DiffusionDITNetVertices(nn.Module):
    def __init__(self, input_channels=385 * 3, hidden_size=1024, n_seed=8, num_layers=8):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_prj = nn.Linear(input_channels, hidden_size)

        self.transformer_blocks = nn.ModuleList([DiTBlock2(hidden_size=hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, input_channels)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(31, 64)
        self.time_style_fuse = nn.Linear(hidden_size, hidden_size)

        self.seed_prj = nn.Linear(input_channels, hidden_size)
        self.n_seed = n_seed

        self.hubert_prj = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.hubert_fuse = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, time_steps, **kwargs):
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        style_embed = self.style_embed(one_hots)
        seed_poses = self.seed_prj(seed_poses[:, :self.n_seed])

        time_style_embed = self.time_style_fuse(torch.cat([time_embed, style_embed], dim=-1))


        
        x = self.in_prj(x)
        x = self.hubert_fuse(torch.cat([x, self.hubert_prj(hubert)], dim=-1))
        # x = torch.cat([seed_poses, x], dim=1)
        x = self.pos_enc(x)
        seed_poses = self.pos_enc(seed_poses)

        kwargs = {
            "time_embed":time_style_embed,
            "seed_poses": seed_poses
        }
        for blk in self.transformer_blocks:
            x = blk(x, **kwargs)

        # x = self.head(x)[:, self.n_seed:]
        x = self.head(x)
        return x
    
class MDMPartsTransCrossSeed(nn.Module):
    def __init__(self, hidden_size=256, n_seed=8, num_layers=8, num_person=31):
        super().__init__()
        self.pos_enc = Summer(PositionalEncoding1D(hidden_size))
        self.in_hand = nn.Linear(30 * 6, hidden_size)
        self.in_upper = nn.Linear(13 * 6, hidden_size)
        self.in_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.in_face = nn.Linear(3 * 6 + 100, hidden_size)
        self.in_x = nn.Linear(55 * 6 + 100 + 3, hidden_size)

        self.hand_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.upper_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.lower_fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.face_fuse = nn.Linear(hidden_size * 2, hidden_size)

        transformer_encoder_layer = nn.TransformerEncoderLayer(hidden_size, 4, hidden_size * 2, batch_first=True)
        self.transformer_hand = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.transformer_upper = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.transformer_lower = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.transformer_face = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        
        self.hand = nn.Linear(hidden_size, 30 * 6)
        self.upper = nn.Linear(hidden_size, 13 * 6)
        self.lower = nn.Linear(hidden_size, 9 * 6 + 3)
        self.face = nn.Linear(hidden_size, 3 * 6 + 100) 

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - 64),
        )
        self.style_embed = StyleEmbedding(num_person, 64)
        self.num_person = num_person

        self.seed_prj_hand = nn.Linear(30 * 6, hidden_size)
        self.seed_prj_upper = nn.Linear(13 * 6, hidden_size)
        self.seed_prj_lower = nn.Linear(9 * 6 + 3, hidden_size)
        self.seed_prj_face = nn.Linear(3 * 6, hidden_size)
        
        self.cross_seed_hand = CrossAttentionLayerWithRoPE(hidden_size, 4)
        self.cross_seed_upper = CrossAttentionLayerWithRoPE(hidden_size, 4)
        self.cross_seed_lower = CrossAttentionLayerWithRoPE(hidden_size, 4)
        self.cross_seed_face = CrossAttentionLayerWithRoPE(hidden_size, 4)  

        self.time_style_hand_fuse = nn.Linear(hidden_size, hidden_size)
        self.time_style_upper_fuse = nn.Linear(hidden_size, hidden_size)
        self.time_style_lower_fuse = nn.Linear(hidden_size, hidden_size)
        self.time_style_face_fuse = nn.Linear(hidden_size, hidden_size)

        self.n_seed = n_seed

        self.hubert_hand_prj = nn.Linear(1024, hidden_size // 2)
        self.hubert_upper_prj = nn.Linear(1024, hidden_size // 2)
        self.hubert_lower_prj = nn.Linear(1024, hidden_size // 2)
        self.hubert_face_prj = nn.Linear(1024, hidden_size // 2)

        self.hubert_hand_fuse = nn.Linear(hidden_size + hidden_size // 2, hidden_size)
        self.hubert_upper_fuse = nn.Linear(hidden_size + hidden_size // 2, hidden_size)
        self.hubert_lower_fuse = nn.Linear(hidden_size + hidden_size // 2, hidden_size)
        self.hubert_face_fuse = nn.Linear(hidden_size + hidden_size // 2, hidden_size)

    def forward(self, x, time_steps, **kwargs):
        hand_indices = kwargs['hand_indices']
        upper_indices = kwargs['upper_indices']
        lower_indices = kwargs['lower_indices']
        face_indices = kwargs['face_indices']
        face_joint_indices = kwargs['face_joint_indices']
        pid = kwargs['pid']
        seed_poses = kwargs['seed_poses']
        hubert = kwargs['hubert']
        
        
        x0 = x
        hand = x[..., hand_indices]
        upper = x[..., upper_indices]
        lower = x[..., lower_indices]
        face = x[..., face_indices]

        hand = self.in_hand(hand)
        upper = self.in_upper(upper)
        lower = self.in_lower(lower)
        face = self.in_face(face)
        x = self.in_x(x)

        hand = self.hand_fuse(torch.cat([hand, x], dim=-1))
        upper = self.upper_fuse(torch.cat([upper, x], dim=-1))
        lower = self.lower_fuse(torch.cat([lower, x], dim=-1))
        face = self.face_fuse(torch.cat([face, x], dim=-1))

        time_embed = self.time_embed(time_steps.reshape((-1, 1)))
        one_hots = F.one_hot(pid, num_classes=31).float().to(x.device)
        style_embed = self.style_embed(one_hots)

        seed_hand_poses = self.seed_prj_hand(seed_poses[:, :self.n_seed, hand_indices])
        seed_upper_poses = self.seed_prj_upper(seed_poses[:, :self.n_seed, upper_indices])
        seed_lower_poses = self.seed_prj_lower(seed_poses[:, :self.n_seed, lower_indices])
        seed_face_poses = self.seed_prj_face(seed_poses[:, :self.n_seed, face_joint_indices])
        
        seed_hand_poses = self.pos_enc(seed_hand_poses)
        seed_upper_poses = self.pos_enc(seed_upper_poses)
        seed_lower_poses = self.pos_enc(seed_lower_poses)
        seed_face_poses = self.pos_enc(seed_face_poses)
        hand_0 = self.pos_enc(hand[:,:self.n_seed])
        upper_0 = self.pos_enc(upper[:,:self.n_seed])
        lower_0 = self.pos_enc(lower[:,:self.n_seed])
        face_0 = self.pos_enc(face[:,:self.n_seed])
        hand_0 = self.cross_seed_hand(hand_0, seed_hand_poses, seed_hand_poses)
        upper_0 = self.cross_seed_upper(upper_0, seed_upper_poses, seed_upper_poses)
        lower_0 = self.cross_seed_lower(lower_0, seed_lower_poses, seed_lower_poses)
        face_0 = self.cross_seed_face(face_0, seed_face_poses, seed_face_poses)
        hand = torch.cat([hand_0, hand[:,self.n_seed:]], dim=1)
        upper = torch.cat([upper_0, upper[:,self.n_seed:]], dim=1)
        lower = torch.cat([lower_0, lower[:,self.n_seed:]], dim=1)
        face = torch.cat([face_0, face[:,self.n_seed:]], dim=1)
        
        

        time_style_hand_embed = self.time_style_hand_fuse(torch.cat([time_embed, style_embed], dim=-1))
        time_style_upper_embed = self.time_style_upper_fuse(torch.cat([time_embed, style_embed], dim=-1))
        time_style_lower_embed = self.time_style_lower_fuse(torch.cat([time_embed, style_embed], dim=-1))
        time_style_face_embed = self.time_style_face_fuse(torch.cat([time_embed, style_embed], dim=-1))

        time_style_hand_embed = time_style_hand_embed.unsqueeze(1)
        time_style_upper_embed = time_style_upper_embed.unsqueeze(1)
        time_style_lower_embed = time_style_lower_embed.unsqueeze(1)
        time_style_face_embed = time_style_face_embed.unsqueeze(1)

        hubert_hand = self.hubert_hand_prj(hubert)
        hubert_upper = self.hubert_upper_prj(hubert)
        hubert_lower = self.hubert_lower_prj(hubert)
        hubert_face = self.hubert_face_prj(hubert)

        hand = self.hubert_hand_fuse(torch.cat([hand, hubert_hand], dim=-1))
        upper = self.hubert_upper_fuse(torch.cat([upper, hubert_upper], dim=-1))
        lower = self.hubert_lower_fuse(torch.cat([lower, hubert_lower], dim=-1))
        face = self.hubert_face_fuse(torch.cat([face, hubert_face], dim=-1))

        hand = torch.cat([time_style_hand_embed,  hand], dim=1)
        upper = torch.cat([time_style_upper_embed, upper], dim=1)
        lower = torch.cat([time_style_lower_embed,  lower], dim=1)
        face = torch.cat([time_style_face_embed,  face], dim=1)

        hand = self.pos_enc(hand)
        upper = self.pos_enc(upper)
        lower = self.pos_enc(lower)
        face = self.pos_enc(face)

        hand = self.transformer_hand(hand)
        upper = self.transformer_upper(upper)
        lower = self.transformer_lower(lower)
        face = self.transformer_face(face)

        hand = self.hand(hand)[:, 1:]
        upper = self.upper(upper)[:, 1:]
        lower = self.lower(lower)[:, 1:]
        face = self.face(face)[:, 1:]

        x = torch.zeros(x0.shape, dtype=hand.dtype, device=hand.device)
        x[..., hand_indices] = hand
        x[..., upper_indices] = upper
        x[..., lower_indices] = lower
        x[..., face_indices] = face
        return x
        
if __name__ == "__main__":
    net = SimpleSpeechModel().cuda()
    huberts = torch.randn(2, 60, 1024).cuda()
    pid = torch.randint(0, 31, (2, )).cuda()
    print(huberts.shape, pid.shape)
    
    r = net(huberts, pid)
    print(r.shape)