
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

import argparse
import os
import sys
sys.path.append('..')
import torch
import random
import numpy as np
from VAE import FrameLevelVAE
from BEATXDataset import LMDBDataset as  BEATXDataset
import time
from einops import rearrange
import torch.nn.functional as F
from Util import axis_angle_to_rotation_6d, rotation_6d_to_axis_angle
from TranslationRegressor.TranslationRegressor import TranslationRegressor
from SMPLX_Util import GlobalToLocalAA


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # 这一句话会使VQVAE的encoder变的很慢


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--test_lmdb_path", type=str,
                        default="../../Data/BEAT2/test_seq_size_60_stride_size_20_global.lmdb")
    parser.add_argument("--seed", type=int, default="0")
    parser.add_argument("--include_translation", type=int, default="0")

    args = parser.parse_args()
    setup_seed(args.seed)

    test_dataset = BEATXDataset(lmdb_path=args.test_lmdb_path)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, num_workers=4, drop_last=True
    )

    net = FrameLevelVAE(
               pred_logvar=True).cuda().requires_grad_(False).eval()
    ckpt = torch.load(args.ckpt)
    net.load_state_dict(ckpt)



    for data in test_dataloader:
        poses = data['poses'].cuda()
        trans = data['trans'].cuda()
        betas = data['betas'].cuda()
        pid = data['pid'].cuda()
        expressions = data['expressions'].cuda()
        print(pid)
        if args.include_translation:
            motion = torch.cat([poses, trans], dim=-1)
        else:
            motion = poses
        with torch.no_grad():
            rec_poses, mu, logvar = net(motion, False)
            if args.include_translation:
                rec_trans = rec_poses[..., -3:]
                rec_poses = rec_poses[..., :-3]
        break


    id = 5
    poses_aa = rotation_6d_to_axis_angle(poses.cpu().numpy())[id]
    rec_poses_aa = rotation_6d_to_axis_angle(rec_poses.cpu().numpy())[id]
    poses_aa = GlobalToLocalAA(poses_aa)
    rec_poses_aa = GlobalToLocalAA(rec_poses_aa)
    np.savez(
        "gt.npz",
        betas=betas[id].cpu().numpy(),
        poses=poses_aa,
        expressions=expressions[id].cpu().numpy(),
        trans=trans[0].cpu().numpy(),
        gender=np.array("neutral"),
        mocap_frame_rate=np.array(30),
        model="smplx2020",
    )


    np.savez(
        "test.npz",
        betas=betas[0].cpu().numpy(),
        poses=rec_poses_aa,
        expressions=expressions[0].cpu().numpy(),
        trans=trans[0].cpu().numpy(),
        gender=np.array("neutral"),
        mocap_frame_rate=np.array(30),
        model="smplx2020",
    )


if __name__ == "__main__":
    main()
