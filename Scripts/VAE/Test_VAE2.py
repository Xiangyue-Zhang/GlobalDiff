
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

import torch
import random
import numpy as np
from VAE import VAE2
from BEATXDataset import LMDBDataset
import time
from einops import rearrange
import torch.nn.functional as F
from Util import rotation_6d_to_axis_angle, NormalizeRot6d, rotation_6d_to_matrix, rotation_matrix_to_axis_angle
from TranslationRegressor.TranslationRegressor import TranslationRegressor
from SMPLX_Util import GetSMPLXModel, SplitPosesIntoSmplxParams, GlobalToLocalAA, global_to_local_batch



def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # 这一句话会使VQVAE的encoder变的很慢


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--trans_ckpt", type=str, default="/mnt/4TDisk/mm_data/mm/project/LargeMotionModel/Scripts/TranslationRegressor/ckpt/best.pt")
    parser.add_argument("--lmdb_path", type=str,
                        default="/mnt/4TDisk/mm_data/mm/data/BEAT/V2/seq_size_60_stride_size_20_global.lmdb")
    parser.add_argument("--seed", type=int, default="0")
    parser.add_argument("--include_translation", type=int, default="0")

    args = parser.parse_args()
    setup_seed(args.seed)

    train_dataset = LMDBDataset(lmdb_path=args.lmdb_path)
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [
                                                                0.8, 0.2])
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, num_workers=4, drop_last=True
    )
    net = VAE2(in_channels=55 * 6 + (3 if args.include_translation else 0), pred_logvar=True).cuda()
    ckpt = torch.load(args.ckpt)
    net.load_state_dict(ckpt)

    smplx_model = GetSMPLXModel().requires_grad_(False).eval().cuda()

    if args.trans_ckpt != "":
        translation_net = TranslationRegressor( ).cuda().requires_grad_(False).eval()
        ckpt = torch.load(args.trans_ckpt)
        translation_net.load_state_dict(ckpt)


    
    data_idx = 0
    for data in test_dataloader:
        poses = data['poses'].cuda()
        trans = data['trans'].cuda()
        betas = data['betas'].cuda()
        pid = data['pid'].cuda()
        expressions = data['expressions'].cuda()

        with torch.no_grad():
            rec, mu, logvar = net(torch.cat([poses, trans], dim=-1), reparameterize=False)
            rec_trans = rec[...,330:]
            rec = rec[...,:330]

        data_idx = data_idx + 1
        if data_idx >=2:
            break

    poses_gt = rotation_6d_to_matrix(poses.reshape((-1, 55, 6)))
    poses_gt = global_to_local_batch(poses_gt.reshape((-1, 55, 3,3))).reshape((1, -1, 55, 3, 3))
    poses_gt = rotation_matrix_to_axis_angle(poses_gt).reshape((-1, 55 * 3)) - smplx_model.pose_mean.cuda()
    
    rec_poses = rotation_6d_to_matrix(rec.reshape((-1, 55, 6)))
    rec_poses = global_to_local_batch(rec_poses.reshape((-1, 55, 3,3))).reshape((1, -1, 55, 3, 3))
    rec_poses = rotation_matrix_to_axis_angle(rec_poses).reshape((-1, 55 * 3)) - smplx_model.pose_mean.cuda()

    np.savez(
        "gt.npz",
        betas=betas[0].cpu().numpy(),
        poses=poses_gt.cpu().numpy(),
        expressions=expressions[0].cpu().numpy(),
        trans=trans[0].cpu().numpy(),
        gender=np.array("neutral"),
        mocap_frame_rate=np.array(30),
        model="smplx2020",
    )


    np.savez(
        "test.npz",
        betas=betas[0].cpu().numpy(),
        poses=rec_poses.cpu().numpy(),
        expressions=expressions[0].cpu().numpy(),
        trans=rec_trans[0].cpu().numpy(),
        gender=np.array("neutral"),
        mocap_frame_rate=np.array(30),
        model="smplx2020",
    )


if __name__ == "__main__":
    main()
