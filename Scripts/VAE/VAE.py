
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
import torch
import sys
sys.path.append("..")
from BasicBlocks import Encoder, Decoder, ResBlock1D, ResBlock1D_2
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer



class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def compute_geodesic_distance(self, m1, m2):
        """Compute the geodesic distance between two rotation matrices.

        Args:
            m1, m2: Two rotation matrices with the shape (batch x 3 x 3).

        Returns:
            The minimal angular difference between two rotation matrices in radian form [0, pi].
        """
        m1 = m1.reshape(-1, 3, 3)
        m2 = m2.reshape(-1, 3, 3)
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.clamp(cos, min=-1 + 1e-6, max=1 - 1e-6)

        theta = torch.acos(cos)

        return theta

    def __call__(self, m1, m2, reduction="mean"):
        loss = self.compute_geodesic_distance(m1, m2)

        if reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            return loss
        else:
            raise RuntimeError(f"unsupported reduction: {reduction}")


def rotation_6d_to_matrix_torch(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class ReconsLoss(nn.Module):
    def __init__(self, rotation_channels, trans_channels, rotation_method="rot6d"):
        super(ReconsLoss, self).__init__()
        self.Loss = torch.nn.L1Loss(reduction="mean")
        self.rec_loss = GeodesicLoss()
        self.rotation_channels = rotation_channels
        self.trans_channels = trans_channels

        self.rotation_method = rotation_method

    def forward(self, motion_pred, motion_gt):
        if self.rotation_method == "rot6d":
            rotation_loss = 0
            motion_pred_r = rotation_6d_to_matrix_torch(motion_pred[:, :, :self.rotation_channels].reshape((-1, 6)))
            motion_gt_r = rotation_6d_to_matrix_torch(motion_gt[:, :, : self.rotation_channels].reshape((-1, 6)))
            rotation_loss = self.rec_loss(motion_pred_r, motion_gt_r)

            trans_loss = 0
            if self.trans_channels > 0:
                motion_pred_t = motion_pred[:, :, self.rotation_channels:]
                motion_gt_t = motion_gt[:, :, self.rotation_channels:]
                trans_loss = F.l1_loss(motion_pred_t, motion_gt_t)
            return rotation_loss + trans_loss
        else:
            return F.smooth_l1_loss(motion_pred, motion_gt)

    def forward_vel(self, motion_pred, motion_gt):  # 1
        loss = self.Loss(
            motion_pred[:, 1:] - motion_pred[:, :-1], motion_gt[:, 1:] - motion_gt[:, :-1])
        return loss

    def forward_smooth(self, motion_pred, motion_gt):

        B, L, C = motion_pred.shape
        pred = motion_pred[:, 0: L - 2] + \
            motion_pred[:, 2:L] - motion_pred[:, 1: L - 1] * 2
        gt = motion_gt[:, 0: L - 2] + \
            motion_gt[:, 2:L] - motion_gt[:, 1: L - 1] * 2
        loss = self.Loss(pred, gt)
        return loss




class VAE(nn.Module):
    def __init__(self,
                 in_channels,
                 model_channels=[256, 512, 512],
                 down_t=2,
                 num_res_per_blocks=2, 
                 kernel_size=[3,3,3],
                 block_type="res1"):
        super().__init__()
        down_block_ids = []
        up_block_ids = []
        if down_t > 0:
            down_block_ids = list(range(0, down_t))
            up_block_ids = list(range(0, down_t))
        self.E = Encoder(in_channels=in_channels,
                         model_channels=model_channels,
                         down_block_ids=down_block_ids,
                         num_res_per_blocks=num_res_per_blocks,
                         down_sample="conv",
                         final_output_channels=model_channels[-1] * 2,
                         kernel_size=kernel_size,
                         block_type=block_type)
        self.D = Decoder(in_channels=model_channels[-1],
                         model_channels=model_channels[::-1],
                         up_block_ids=up_block_ids,
                         num_res_per_block=num_res_per_blocks,
                         up_sample="conv",
                         final_output_channels=in_channels,
                         kernel_size=kernel_size,
                         block_type=block_type)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, reparameterize=True):
        mu, logvar = self.E(x).chunk(2, dim=1)

        if reparameterize:
            z = self.reparameterize(mu, logvar=logvar)
        else:
            z = mu
        rec = self.D(z)
        return rec, mu, logvar

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim=256, pred_logvar=False):
        super().__init__()
        self.in_prj = nn.Linear(input_dim, hidden_dim)
        self.pe = Summer(PositionalEncoding1D(hidden_dim))
        self.resnets = nn.Sequential(
            ResBlock1D(hidden_dim, hidden_dim),
            ResBlock1D(hidden_dim, hidden_dim),
            ResBlock1D(hidden_dim, hidden_dim),
            nn.MaxPool1d(2, stride=2),
        )
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True, norm_first=False)
        self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.out_prj = nn.Linear(hidden_dim, latent_dim if not pred_logvar  else latent_dim * 2)

    def forward(self, x):
        B, L = x.shape[:2]
        x = self.in_prj(x)
        x = rearrange(x, "b l c -> b c l")
        x = self.resnets(x)
        x = rearrange(x, "b c l -> b l c")
        x = self.pe(x)
        x = self.encoders(x)
        x = self.out_prj(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim=256):
        super().__init__()
        self.in_prj = nn.Linear(latent_dim, hidden_dim)
        self.pe = Summer(PositionalEncoding1D(hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True, norm_first=False)
        self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.resnets = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ResBlock1D(hidden_dim, hidden_dim),
            ResBlock1D(hidden_dim, hidden_dim),
            ResBlock1D(hidden_dim, hidden_dim),
        )
        self.out_prj = nn.Linear(hidden_dim, input_dim)
        

    def forward(self, z):
        z = self.in_prj(z)
        z  =self.pe(z)
        z = self.encoders(z)
        z = rearrange(z, "b l c -> b c l")
        z = self.resnets(z)
        z = rearrange(z, "b c l -> b l c")
        x = self.out_prj(z)
        return x


class VAE2(nn.Module):
    def __init__(self, in_channels=330, hidden_size = 256, pred_logvar=False):
        super().__init__()
        self.E = TransformerEncoder(in_channels, hidden_size, latent_dim=128, pred_logvar=pred_logvar)
        self.D = TransformerDecoder(in_channels, hidden_size, latent_dim=128)
        self.pred_logvar = pred_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, reparameterize=True):
        if self.pred_logvar:
            mu, logvar = self.E(x).chunk(2, dim=-1)
        else:
            mu = self.E(x)
            logvar = None

        if reparameterize:
            assert self.pred_logvar
            z = self.reparameterize(mu, logvar=logvar)
        else:
            z = mu
        rec = self.D(z)
        return rec, mu, logvar


class MaskedVAE2(nn.Module):
    def __init__(self, in_channels=330, hidden_size = 256, pred_logvar=False):
        super().__init__()
        self.E = TransformerEncoder(hidden_size, hidden_size, latent_dim=128, pred_logvar=pred_logvar)
        self.D = TransformerDecoder(in_channels, hidden_size, latent_dim=128)
        self.pred_logvar = pred_logvar

        self.in_prj = nn.Linear(in_channels, hidden_size)
        self.mask_val = nn.Parameter(torch.randn(1, 1, hidden_size))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):

        x = self.in_prj(x)

        mu = self.E(x)
        return mu

    def forward(self, x, mask=None, reparameterize=True):
        """ Args:
            x: (B, L, C)
            mask: (B, L)
            reparameterize: bool, whether to reparameterize the latent variable
        """

        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_val, self.in_prj(x))
        else:
            x = self.in_prj(x)
        if self.pred_logvar:
            mu, logvar = self.E(x).chunk(2, dim=-1)
        else:
            mu = self.E(x)
            logvar = None

        if reparameterize:
            assert self.pred_logvar
            z = self.reparameterize(mu, logvar=logvar)
        else:
            z = mu
        rec = self.D(z)
        return rec, mu, logvar


class VAE3(nn.Module):
    class TransformerEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim=256, pred_logvar=False):
            super().__init__()
            self.in_prj = nn.Linear(input_dim, hidden_dim)
            self.pe = Summer(PositionalEncoding1D(hidden_dim))
            self.resnets = nn.Sequential(
                ResBlock1D(hidden_dim, hidden_dim),
                ResBlock1D(hidden_dim, hidden_dim),
                nn.MaxPool1d(2, stride=2),
                ResBlock1D(hidden_dim, hidden_dim),
                ResBlock1D(hidden_dim, hidden_dim),
                nn.MaxPool1d(2, stride=2),
            )
            encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True, norm_first=False)
            self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.out_prj = nn.Linear(hidden_dim, latent_dim if not pred_logvar  else latent_dim * 2)

        def forward(self, x):
            B, L = x.shape[:2]
            x = self.in_prj(x)
            x = rearrange(x, "b l c -> b c l")
            x = self.resnets(x)
            x = rearrange(x, "b c l -> b l c")
            x = self.pe(x)
            x = self.encoders(x)
            x = self.out_prj(x)
            return x
        
    class TransformerDecoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim=256):
            super().__init__()
            self.in_prj = nn.Linear(latent_dim, hidden_dim)
            self.pe = Summer(PositionalEncoding1D(hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True, norm_first=False)
            self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.resnets = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ResBlock1D(hidden_dim, hidden_dim),
                ResBlock1D(hidden_dim, hidden_dim),
                nn.Upsample(scale_factor=2),
                ResBlock1D(hidden_dim, hidden_dim),
                ResBlock1D(hidden_dim, hidden_dim),
            )
            self.out_prj = nn.Linear(hidden_dim, input_dim)
            

        def forward(self, z):
            z = self.in_prj(z)
            z  =self.pe(z)
            z = self.encoders(z)
            z = rearrange(z, "b l c -> b c l")
            z = self.resnets(z)
            z = rearrange(z, "b c l -> b l c")
            x = self.out_prj(z)
            return x
        
    def __init__(self, in_channels=330, hidden_size = 256, pred_logvar=False):
        super().__init__()
        self.E = VAE3.TransformerEncoder(in_channels, hidden_size, latent_dim=128, pred_logvar=pred_logvar)
        self.D = VAE3.TransformerDecoder(in_channels, hidden_size, latent_dim=128)
        self.pred_logvar = pred_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, reparameterize=True):
        if self.pred_logvar:
            mu, logvar = self.E(x).chunk(2, dim=-1)
        else:
            mu = self.E(x)
            logvar = None

        if reparameterize:
            assert self.pred_logvar
            z = self.reparameterize(mu, logvar=logvar)
        else:
            z = mu
        rec = self.D(z)
        return rec, mu, logvar


class MaskedVAE3(nn.Module):
    class TransformerEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim=256, pred_logvar=False):
            super().__init__()
            self.pe = Summer(PositionalEncoding1D(hidden_dim))
            self.resnets = nn.Sequential(
                ResBlock1D(hidden_dim, hidden_dim),
                ResBlock1D(hidden_dim, hidden_dim),
                nn.MaxPool1d(2, stride=2),
                ResBlock1D(hidden_dim, hidden_dim),
                ResBlock1D(hidden_dim, hidden_dim),
                nn.MaxPool1d(2, stride=2),
            )
            encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True, norm_first=False)
            self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.out_prj = nn.Linear(hidden_dim, latent_dim if not pred_logvar  else latent_dim * 2)

            self.in_prj = nn.Linear(input_dim, hidden_dim)
        def forward(self, x):
            B, L = x.shape[:2]
            x = self.in_prj(x)
            x = rearrange(x, "b l c -> b c l")
            x = self.resnets(x)
            x = rearrange(x, "b c l -> b l c")
            x = self.pe(x)
            x = self.encoders(x)
            x = self.out_prj(x)
            return x
        
    class TransformerDecoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim=256):
            super().__init__()
            self.in_prj = nn.Linear(latent_dim, hidden_dim)
            self.pe = Summer(PositionalEncoding1D(hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True, norm_first=False)
            self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.resnets = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ResBlock1D(hidden_dim, hidden_dim),
                ResBlock1D(hidden_dim, hidden_dim),
                nn.Upsample(scale_factor=2),
                ResBlock1D(hidden_dim, hidden_dim),
                ResBlock1D(hidden_dim, hidden_dim),
            )
            self.out_prj = nn.Linear(hidden_dim, input_dim)
            

        def forward(self, z):
            z = self.in_prj(z)
            z  =self.pe(z)
            z = self.encoders(z)
            z = rearrange(z, "b l c -> b c l")
            z = self.resnets(z)
            z = rearrange(z, "b c l -> b l c")
            x = self.out_prj(z)
            return x
        
    def __init__(self, in_channels=330, hidden_size = 256, pred_logvar=False):
        super().__init__()
        self.E = VAE3.TransformerEncoder(hidden_size, hidden_size, latent_dim=128, pred_logvar=pred_logvar)
        self.D = VAE3.TransformerDecoder(in_channels, hidden_size, latent_dim=128)
        self.pred_logvar = pred_logvar

        self.in_prj = nn.Linear(in_channels, hidden_size)
        self.mask_val = nn.Parameter(torch.randn(1, 1, hidden_size))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):

        x = self.in_prj(x)

        mu = self.E(x)
        return mu

    def forward(self, x, mask=None, reparameterize=True):
        '''
        Args:            
            x: (B, L, C)
            mask: (B, L), boolean mask
        '''
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_val, self.in_prj(x))
        else:
            x = self.in_prj(x)
        if self.pred_logvar:
            mu, logvar = self.E(x).chunk(2, dim=-1)
        else:
            mu = self.E(x)
            logvar = None

        if reparameterize:
            assert self.pred_logvar
            z = self.reparameterize(mu, logvar=logvar)
        else:
            z = mu
        rec = self.D(z)
        return rec, mu, logvar

class VAE4(nn.Module):
    class TransformerEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim=256, pred_logvar=False):
            super().__init__()
            self.in_prj = nn.Linear(input_dim, hidden_dim)
            self.pe = Summer(PositionalEncoding1D(hidden_dim))
            self.resnets = nn.Sequential(
                ResBlock1D(hidden_dim, hidden_dim),
                ResBlock1D(hidden_dim, hidden_dim),
            )
            encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True, norm_first=False)
            self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.out_prj = nn.Linear(hidden_dim, latent_dim if not pred_logvar  else latent_dim * 2)

        def forward(self, x):
            B, L = x.shape[:2]
            x = self.in_prj(x)
            x = rearrange(x, "b l c -> b c l")
            x = self.resnets(x)
            x = rearrange(x, "b c l -> b l c")
            x = self.pe(x)
            x = self.encoders(x)
            x = self.out_prj(x)
            return x
        
    class TransformerDecoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim=256):
            super().__init__()
            self.in_prj = nn.Linear(latent_dim, hidden_dim)
            self.pe = Summer(PositionalEncoding1D(hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True, norm_first=False)
            self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.resnets = nn.Sequential(
                ResBlock1D(hidden_dim, hidden_dim),
                ResBlock1D(hidden_dim, hidden_dim),
            )
            self.out_prj = nn.Linear(hidden_dim, input_dim)
            

        def forward(self, z):
            z = self.in_prj(z)
            z  =self.pe(z)
            z = self.encoders(z)
            z = rearrange(z, "b l c -> b c l")
            z = self.resnets(z)
            z = rearrange(z, "b c l -> b l c")
            x = self.out_prj(z)
            return x
        
    def __init__(self, in_channels=330, hidden_size = 256, latent_dim=128, pred_logvar=False):
        super().__init__()
        self.E = VAE4.TransformerEncoder(in_channels, hidden_size, latent_dim=latent_dim, pred_logvar=pred_logvar)
        self.D = VAE4.TransformerDecoder(in_channels, hidden_size, latent_dim=latent_dim)
        self.pred_logvar = pred_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, reparameterize=True):
        if self.pred_logvar:
            mu, logvar = self.E(x).chunk(2, dim=-1)
        else:
            mu = self.E(x)
            logvar = None

        if reparameterize:
            assert self.pred_logvar
            z = self.reparameterize(mu, logvar=logvar)
        else:
            z = mu
        rec = self.D(z)
        return rec, mu, logvar

class VAE6(nn.Module):
    class TransformerEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim=256, pred_logvar=False):
            super().__init__()
            self.in_prj = nn.Linear(input_dim, hidden_dim)
            self.pe = Summer(PositionalEncoding1D(hidden_dim))
            self.resnets = nn.Sequential(
                ResBlock1D(hidden_dim, hidden_dim),
                ResBlock1D(hidden_dim, hidden_dim),
                nn.Upsample(scale_factor=2),
            )
            encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True, norm_first=False)
            self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.out_prj = nn.Linear(hidden_dim, latent_dim if not pred_logvar  else latent_dim * 2)

        def forward(self, x):
            B, L = x.shape[:2]
            x = self.in_prj(x)
            x = rearrange(x, "b l c -> b c l")
            x = self.resnets(x)
            x = rearrange(x, "b c l -> b l c")
            x = self.pe(x)
            x = self.encoders(x)
            x = self.out_prj(x)
            return x
        
    class TransformerDecoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim=256):
            super().__init__()
            self.in_prj = nn.Linear(latent_dim, hidden_dim)
            self.pe = Summer(PositionalEncoding1D(hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True, norm_first=False)
            self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.resnets = nn.Sequential(
                ResBlock1D(hidden_dim, hidden_dim),
                ResBlock1D(hidden_dim, hidden_dim),
                nn.MaxPool1d(2, stride=2),
            )
            self.out_prj = nn.Linear(hidden_dim, input_dim)
            

        def forward(self, z):
            z = self.in_prj(z)
            z  =self.pe(z)
            z = self.encoders(z)
            z = rearrange(z, "b l c -> b c l")
            z = self.resnets(z)
            z = rearrange(z, "b c l -> b l c")
            x = self.out_prj(z)
            return x
        
    def __init__(self, in_channels=330, hidden_size = 256, pred_logvar=False):
        super().__init__()
        self.E = VAE6.TransformerEncoder(in_channels, hidden_size, latent_dim=128, pred_logvar=pred_logvar)
        self.D = VAE6.TransformerDecoder(in_channels, hidden_size, latent_dim=128)
        self.pred_logvar = pred_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, reparameterize=True):
        if self.pred_logvar:
            mu, logvar = self.E(x).chunk(2, dim=-1)
        else:
            mu = self.E(x)
            logvar = None

        if reparameterize:
            assert self.pred_logvar
            z = self.reparameterize(mu, logvar=logvar)
        else:
            z = mu
        rec = self.D(z)
        return rec, mu, logvar



class FrameLevelVAE(nn.Module):
    class Encoder(nn.Module):
        def __init__(self,  hidden_dim=384, latent_dim=128, pred_logvar=False):
            super().__init__()
            self.in_prj = nn.Linear(6, hidden_dim)
            self.net = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim*2, batch_first=True, norm_first=True), num_layers=4)
            self.pos_enc = Summer(PositionalEncoding1D(hidden_dim))
            self.out_prj = nn.Linear(hidden_dim, latent_dim * 2 if pred_logvar else latent_dim) 

        def forward(self, x):
            origin_shape = x.shape
            x = self.in_prj(x.reshape(-1, 55, 6))
            x = self.pos_enc(x)
            x = self.net.forward(x)
            x = torch.mean(x, dim=1)
            x = self.out_prj(x)
            x = x.reshape(origin_shape[:-1]  + (-1,))
            return x
        
    # class Decoder(nn.Module):
    #     def __init__(self,  hidden_dim=384, latent_dim=128):
    #         super().__init__()
    #         self.hidden_dim = hidden_dim
    #         self.in_prj = nn.Sequential(
    #             nn.Linear(latent_dim, hidden_dim ),
    #             nn.ReLU(),
    #             nn.Linear(hidden_dim, hidden_dim ),
    #             nn.ReLU(),
    #             nn.Linear(hidden_dim , hidden_dim * 55)
    #         )
    #         self.net = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim*2, batch_first=True, norm_first=True), num_layers=4)
    #         self.pos_enc = Summer(PositionalEncoding1D(hidden_dim))
    #         self.out_prj = nn.Linear(hidden_dim, 6) 

    #     def forward(self, x):
    #         origin_shape = x.shape
    #         x = self.in_prj(x).reshape((-1, 55, self.hidden_dim))
    #         x = self.pos_enc(x)
    #         x = self.net.forward(x)
    #         x = self.out_prj(x)
    #         x = x.reshape(origin_shape[:-1]  + (55 * 6,))
    #         return x
        
    class Decoder(nn.Module):
        def __init__(self,  hidden_dim=384, latent_dim=128):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.in_prj = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim ),
                nn.ReLU(),
                nn.Linear(hidden_dim , hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim ),
                nn.ReLU(),
                nn.Linear(hidden_dim , 55 * 6)
            )


        def forward(self, x):
            x = self.in_prj(x)
            return x

        
    def __init__(self, hidden_size = 256, latent_dim=128, pred_logvar=True):
        super().__init__()
        self.E = FrameLevelVAE.Encoder(hidden_size, latent_dim=latent_dim, pred_logvar=pred_logvar)
        self.D = FrameLevelVAE.Decoder(hidden_size * 2, latent_dim=latent_dim)
        self.pred_logvar = pred_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, reparameterize=True):
        if self.pred_logvar:
            mu, logvar = self.E(x).chunk(2, dim=-1)
        else:
            mu = self.E(x)
            logvar = None

        if reparameterize:
            assert self.pred_logvar
            z = self.reparameterize(mu, logvar=logvar)
        else:
            z = mu
        rec = self.D(z)
        return rec, mu, logvar

if __name__ == '__main__':
    x = torch.randn((1, 60, 330))
    net = FrameLevelVAE(in_channels=330, hidden_size=256, latent_dim=128, pred_logvar=True)
    rec, mu, logvar = net(x)
    print(rec.shape, mu.shape, logvar.shape)
