
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

# python Test_repaint.py /home/mm/Downloads/epoch_999 /media/mm/ssd3T/data/audio/tongyi/xinchunjiajie.wav --data_id=3  --cfg=1.1 --L0=5
import time
import sys
sys.path.append("..")
import torch
import argparse
import librosa
import torch.nn as nn

from BEATXDataset import LMDBDataset
from Util import *
import random


from Util import rotation_6d_to_axis_angle, SplitIntoOverlapedWindows
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
import os
from EMAGE_VAE.test_vae import GetVAE

import numpy as np
from scipy.linalg import sqrtm


def calculate_fid(real_features, generated_features):
    """
    计算真实数据和生成数据之间的FID分数
    
    参数:
    real_features (numpy.ndarray): 真实数据的特征，形状为(N, 256)
    generated_features (numpy.ndarray): 生成数据的特征，形状为(M, 256)
    
    返回:
    float: FID分数
    """
    # 计算均值
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    
    # 计算协方差矩阵
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    # 计算均值差的平方和
    ssdiff = np.sum((mu_real - mu_gen) ** 2.0)
    
    # 计算协方差矩阵的平方根乘积的迹
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    
    # 确保covmean是实数（由于数值误差可能有微小虚部）
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # 计算FID
    fid = ssdiff + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
    
    return fid

def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="test_out")
    parser.add_argument("--gt_folder", type=str, default="/mnt/4TDisk/mm_data/mm/data/BEAT/V2/beat_v2.0.0/beat_english_v2.0.0/smplxflame_30")
    parser.add_argument("--seed", type=int, default="0")
    parser.add_argument("--outfile", type=str, default="")
    args = parser.parse_args()

    setup_seed(0)

    npz_files = glob(os.path.join(args.data_folder, "*.npz"))
    test_data_lst = []
    gt_data_lst = []
    
    expressions = 0
    num_expressions = 0
    prefix = "res_"
    for npz_file in tqdm(npz_files):
        # smplx coefficients
        base_name = os.path.basename(npz_file)
        if base_name.startswith(prefix):
            smplx_data = np.load(npz_file, allow_pickle=True)
            poses = smplx_data["poses"].astype(np.float32).reshape((-1, 55 * 3))
            trans = smplx_data["trans"].astype(np.float32)
            expressions_pred = smplx_data["expressions"].astype(np.float32)
            data = np.concatenate([poses, trans], axis=-1).astype(np.float32)
            data = SplitIntoOverlapedWindows(data, 64, 20)
            test_data_lst = test_data_lst + data
        
            gt_npy_file = os.path.join(args.gt_folder, base_name.replace(prefix, ""))
            smplx_data = np.load(gt_npy_file, allow_pickle=True)
            poses = smplx_data["poses"].astype(np.float32)
            trans = smplx_data["trans"].astype(np.float32)
            betas = smplx_data["betas"].astype(np.float32)
            expressions_gt = smplx_data["expressions"].astype(np.float32)
            data = np.concatenate([poses, trans], axis=-1).astype(np.float32)
            data = SplitIntoOverlapedWindows(data, 64, 20)
            gt_data_lst = gt_data_lst + data
            
            min_len = min(len(expressions_pred), len(expressions_gt))
            expressions += np.mean(np.sum((expressions_pred[:min_len] - expressions_gt[:min_len]) ** 2, axis=-1))
            num_expressions += 1
            
    print("Average expression error: ", expressions / num_expressions)
    dataset = [(a, b) for a, b in zip(test_data_lst, gt_data_lst)]
    test_dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0) # default 1024
    
    
    emage_vae = GetVAE()

    summed_fid = 0
    num_batch = 0
    for data in tqdm(test_dataloader):
        num_batch += 1
        pred, gt = data[0].cuda(), data[1].cuda()
        pred_rot6d = axis_angle_to_rotation_6d(pred[..., :165])
        gt_rot6d = axis_angle_to_rotation_6d(gt[..., :165])
        pred = torch.cat([pred_rot6d, pred[..., 165:]], dim=-1)
        gt = torch.cat([gt_rot6d, gt[..., 165:]], dim=-1)

        pred_mu = emage_vae.map2latent(pred[...,:330])
        gt_mu = emage_vae.map2latent(gt[...,:330])
        # pred_mu, pred_logvar = vae.E(pred).chunk(2, dim=-1)
        # gt_mu, gt_logvar = vae.E(gt).chunk(2, dim=-1)

        latent_dim = gt_mu.shape[-1]
        fid = calculate_fid(gt_mu.cpu().numpy().reshape((-1, latent_dim)), pred_mu.cpu().numpy().reshape((-1, latent_dim)))
        # print("FID: ", fid)
        summed_fid += fid

        


    print("Average FID: ", summed_fid / num_batch, "#num_batch", num_batch)
    if args.outfile != "":
        with open(args.outfile, "a") as f:
            f.write(str(summed_fid / num_batch) +"\n")



    # np.savez(
    #     "test.npz",
    #     betas=betas[0].cpu().numpy(),
    #     poses=pred_poses[0].cpu().numpy(),
    #     # expressions=np.zeros((len(sample), 100)),
    #     expressions=pred_expressions[0].cpu().numpy(),
    #     trans=pred_trans[0].cpu().numpy(),
    #     # trans=np.zeros((len(sample), 3)),
    #     gender=np.array("neutral"),
    #     mocap_frame_rate=np.array(30),
    #     model="smplx2020",
    # )


if __name__ == '__main__':
    main()
