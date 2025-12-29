
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

import sys
sys.path.append("..")
import numpy as np
import random
import torch.distributed as dist
import torch.utils.data.distributed
import torch
import os
import argparse
from einops import rearrange
from BEATXDataset import LMDBDataset
from Net import  DiffusionDITNetPartsFixedExpressions2PostNorm as  MotionModel
from Net import SimpleSpeechModel
import time
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from VAE.VAE import ReconsLoss
from SMPLX_Util import GetSMPLXModel, SplitPosesIntoSmplxParams
from Util import rotation_6d_to_axis_angle, rotation_6d_to_matrix, axis_angle_to_rotation_6d
from SMPLXConfig import GetBodyPartIndices
from losses import ReconsLoss, PosesReconsLoss
from FK import GFKWraper
# from VAE.VAE import VAE3, VAE2
from VAE.VAE import MaskedVAE2, MaskedVAE3
from SMPLX_Util import SMPLX_PARENTS
from scipy.io import loadmat

def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
def WriteObj(filename, verts, faces=None):
    with open(filename, 'w') as fh:
        for v in verts:
            fh.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        if faces is not None:
            for f in faces:
                fh.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

def SimilarityStructureLoss(pred: torch.Tensor, target: torch.Tensor):
    assert pred.shape == target.shape
    assert len(pred.shape) == 3
    B, L, D = pred.shape
    eps = 1e-6
    normalized_pred = pred / (torch.norm(pred, dim=-1, keepdim=True) + eps)
    normalized_target = target / (torch.norm(target, dim=-1, keepdim=True) + eps)
    pred_mat = torch.einsum('b i k, b j k-> b i j', normalized_pred, normalized_pred)
    target_mat = torch.einsum('b i k, b j k-> b i j', normalized_target, normalized_target)
    loss = torch.abs(pred_mat - target_mat)
    
    return loss.mean()

def GetSmplxVerts(x, smplx_model, expressions, betas_input=None):
    """
    x:B,T,55 * 3
    """
    B, T, _ = x.shape
    smplx_params = SplitPosesIntoSmplxParams(x)
    for k in smplx_params.keys():
        smplx_params[k] = smplx_params[k].reshape((B * T, -1))
    # betas = betas.reshape((B, 1, -1)).repeat((1, T, 1)).reshape((B * T, -1))
    betas = torch.zeros((B * T, 300), device=x.device) if betas_input is None else betas_input.reshape((B * T, -1))
    output = smplx_model(betas=betas,
                         global_orient=smplx_params["global_orient"],
                         body_pose=smplx_params["body_pose"],
                         expression=expressions.reshape((B * T, 100)),
                         jaw_pose=smplx_params["jaw_pose"],
                         leye_pose=smplx_params["leye_pose"],
                         reye_pose=smplx_params["reye_pose"],
                         left_hand_pose=smplx_params["left_hand_pose"],
                         right_hand_pose=smplx_params["right_hand_pose"],
                         transl=torch.zeros((B * T, 3), device=x.device),
                         return_verts=True)
    vertices = output.vertices.reshape((B, T, -1))
    return vertices


def FlatMSE(pred: torch.Tensor, target: torch.Tensor):
    return torch.mean((pred - target) ** 2, dim=list(range(1, len(pred.shape))))


def VEL(x: torch.Tensor):
    return x[:, 1:] - x[:, :-1]


def ACC(x: torch.Tensor):
    return VEL(VEL(x))


def foot_contact_loss_func(pred_vertices, vertices, pid, foot_contact_data):
    B, L  = pred_vertices.shape[:2]
    left_ankle = vertices[:,:,7,[1]]
    right_ankle = vertices[:,:,8,[1]]
    left_foot = vertices[:,:,10,[1]]
    rigth_foot = vertices[:,:,11,[1]]
    foot_contact_label = torch.cat([left_foot, left_ankle, rigth_foot, right_ankle], dim=-1) < foot_contact_data[pid].unsqueeze(1) + 0.01


    left_foot_contact_label = torch.sum(foot_contact_label[...,:2], dim=-1) == 2
    right_foot_contact_label = torch.sum(foot_contact_label[...,2:], dim=-1) == 2
    left_foot_contact_label = left_foot_contact_label.reshape((B*L,))
    right_foot_contact_label = right_foot_contact_label.reshape((B*L,))
    # print(foot_contact_label.sum() / foot_contact_label.numel())
    foot_contact_loss = 0

    pred_left_foot, left_foot = pred_vertices[:,:,[7,10]], vertices[:,:,[7,10]]
    pred_right_foot, right_foot = pred_vertices[:,:,[8,11]],vertices[:,:,[8,11]]

    if torch.any(left_foot_contact_label):
        foot_contact_loss = foot_contact_loss + F.smooth_l1_loss(pred_left_foot.reshape((B*L, -1, 3))[left_foot_contact_label], left_foot.reshape((B*L, -1, 3))[left_foot_contact_label], beta=0.02)
    if torch.any(right_foot_contact_label):
        foot_contact_loss = foot_contact_loss + F.smooth_l1_loss(pred_right_foot.reshape((B*L, -1, 3))[right_foot_contact_label], right_foot.reshape((B*L, -1, 3))[right_foot_contact_label], beta=0.02)

    return foot_contact_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str,
                        default="../../Data/BEAT2/train_seq_size_60_stride_size_20_global.lmdb")
    parser.add_argument("--lmdb_hubert", type=str,
                        default="../../Data/BEAT2/train_seq_size_60_stride_size_20_global_wavlm.lmdb")

    parser.add_argument("--ckpt_folder", type=str, default="ckpt/split/DiffusionDITNetPartsFixedExpressions2PostNorm_LL_split_HorizonFlip_MaskedVAE3_W02_BoneDirLoss")
    parser.add_argument("--batch_size", type=int, default="128")
    parser.add_argument("--num_workers", type=int, default="4")
    parser.add_argument("--epoch", type=int, default="1000")
    parser.add_argument("--lr", type=float, default="1e-4", help="i use 4e-5 previously")
    parser.add_argument("--local_rank", type=int,
                        help="local rank, will passed by ddp")
    parser.add_argument("--local-rank", type=int,
                        help="local rank, will passed by ddp")
    parser.add_argument("--sched_step", type=int, default="1000")
    parser.add_argument("--save_n_epoch", type=int, default="20")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default="0")
    parser.add_argument("--rotation_method", type=str, default="rot6d")
    parser.add_argument("--classifier_free_prob", type=float, default="0.1")
    parser.add_argument("--sem", type=int, default='1')
    parser.add_argument("--num_encoders", type=int, default='6')
    parser.add_argument('--data_id', type=int, default='0')
    parser.add_argument('--ignore_mel', type=int, default='0')
    parser.add_argument('--fp16', type=int, default='0')
    parser.add_argument('--wav2vec', type=str, default='/mnt/4TDisk/mm_data/mm/huggingface/facebook/hubert-large-ls960-ft')
    args = parser.parse_args()
    setup_seed(args.seed)



    
    
    if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] is not None:
        print(os.environ["LOCAL_RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl", rank=args.local_rank, init_method="env://")

    # vae3 = VAE3(55 * 6 + 3, pred_logvar=True).cuda().requires_grad_(False).eval()
    # # vae3.load_state_dict(torch.load("../VAE/ckpt/split/global/VAE3/best.pt"))
    # vae3.load_state_dict(torch.load("../VAE/ckpt/split/global/VAE3-HorizonFlip/best.pt"))
    
    # vae2= VAE2(55 * 6 + 3, pred_logvar=True).cuda().requires_grad_(False).eval()
    # # vae2.load_state_dict(torch.load("../VAE/ckpt/split/global/VAE2/best.pt"))
    # vae2.load_state_dict(torch.load("../VAE/ckpt/split/global/VAE2-HorizonFlip/best.pt"))


    vae3 = MaskedVAE3(55 * 6 + 3, pred_logvar=True).cuda().requires_grad_(False).eval()
    # vae3.load_state_dict(torch.load("../VAE/ckpt/split/global/VAE3/best.pt"))
    vae3.load_state_dict(torch.load("../VAE/ckpt/split/global/MaskedVAE3-HorizonFlip/best.pt"))
    
    vae2= MaskedVAE2(55 * 6 + 3, pred_logvar=True).cuda().requires_grad_(False).eval()
    # vae2.load_state_dict(torch.load("../VAE/ckpt/split/global/VAE2/best.pt"))
    vae2.load_state_dict(torch.load("../VAE/ckpt/split/global/MaskedVAE2-HorizonFlip/best.pt"))

    # audio_processor = Wav2Vec2Processor.from_pretrained(args.wav2vec)
    # audio_preprocessor = lambda x: audio_processor(x, sampling_rate=16000,return_tensors="pt").input_values.squeeze(0)
    train_dataset = LMDBDataset(lmdb_path=args.lmdb_path, lmdb_hubert=args.lmdb_hubert, rotation_method="axisangle", use_symmetry_aug=False) 
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, 
        prefetch_factor=10, drop_last=True, pin_memory=True, persistent_workers=True
    )

    # net = DiffusionDITNet(input_channels=55 * 6, hidden_size=1024).cuda()
    n_seed = 8
    net = MotionModel(n_seed=n_seed, num_person=31).cuda() 
    GFK = GFKWraper().cuda()
    simple_speech_model = SimpleSpeechModel().requires_grad_(False).eval().cuda()
    simple_speech_model.load_state_dict(torch.load("ckpt/split/SimpleSpeechModel/best.pt"))
    if args.resume != "":
        ckpt = torch.load(args.resume)
        net.load_state_dict(ckpt)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank], find_unused_parameters=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)


    if dist.get_rank() == 0:
        log_dir = os.path.join(args.ckpt_folder, "log")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    hand_indices = GetBodyPartIndices("hand", "rot6d")
    upper_indices = GetBodyPartIndices("upper", "rot6d")
    lower_indices = GetBodyPartIndices("lower", "rot6d")
    trans_indices = np.arange(55 * 6, 55 *6 + 3)
    lower_indices = np.concatenate([lower_indices, trans_indices], axis=0)
    face_joint_indices = GetBodyPartIndices("face", "rot6d")
    expression_indices = np.arange(55 * 6 + 3, 55 *6 + 3 + 100)
    face_indices = np.concatenate([face_joint_indices, expression_indices], axis=0)

        
    pose_loss_func = ReconsLoss(55 * 6, 3)
    pose_loss_func2 = ReconsLoss(55 * 6, 3)

    for epoch in range(args.epoch):
        net.train()
        if dist.get_rank() == 0:
            epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)
        batch_loss = 0
        batch_num = 0
        for batch_idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            poses = data['poses'].cuda()
            poses = axis_angle_to_rotation_6d(poses)
            pid = data['pid'].cuda()
            betas = data['betas'].cuda()
            huberts = data['hubert'].cuda()
            # expressions = data['expressions'].cuda()
            trans = data['trans'].cuda()
            bone_length = data['bone_length'].cuda()
            rest_joints = data['J'].cuda().unsqueeze(1).expand((-1, poses.shape[1], -1, -1)).reshape((-1, 55,  3))

            poses_trans = torch.cat([poses, trans], dim=-1)


            x1 = poses_trans
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=x1.device)
            sigma =  1e-3
            xt = x0 * (1 - t[:,None, None] + sigma * t[:,None, None]) + x1 * t[:,None, None]
            seed_poses = poses_trans[:, :n_seed,:330 + 3]
            if random.random() < args.classifier_free_prob:
                pid = torch.zeros_like(pid)
            pid_for_speech = pid.clone()
            pid_flag = pid > 30
            if torch.any(pid_flag):
                pid_for_speech[pid_flag] = pid_for_speech[pid_flag] - 30
            expressions = simple_speech_model(huberts, pid=pid_for_speech)
            model_kwargs = {"pid": pid,
                            "seed_poses": seed_poses,
                            "hand_indices": hand_indices,
                            "upper_indices": upper_indices,
                            "lower_indices": lower_indices,
                            "face_indices": face_indices,
                            "face_joint_indices": face_joint_indices,
                            "hubert": huberts,
                            "bone_length": bone_length,
                            "expressions": expressions
                            }

            pred_x1 = net(xt, t, **model_kwargs)

            # mse loss
            mse_loss = F.mse_loss(pred_x1, x1)


            geodesic_loss = pose_loss_func(pred_x1, x1) * 1e-3  # 1e-3 for geodesic  and 1e-2 for poserec
            # geodesic_loss = 0
            # geometric loss
            pred_trans = pred_x1[..., 330:333]
            B, L = pred_trans.shape[:2]
            gt_posemat = rotation_6d_to_matrix(poses[...,:330].reshape((-1, 55, 6)))
            pred_posemat = rotation_6d_to_matrix(pred_x1[...,:330].reshape((-1, 55, 6)))
            vertices = GFK(gt_posemat, rest_joints).reshape((B, L, -1, 3)) + trans.unsqueeze(-2)
            pred_vertices = GFK(pred_posemat, rest_joints).reshape((B, L, -1, 3)) + pred_trans.unsqueeze(-2)
            # vertices = vertices[:,:, 55:]
            # pred_vertices = pred_vertices[:,:, 55:]
            vertices_loss = F.smooth_l1_loss(pred_vertices, vertices, beta=0.02)
            vel_loss = F.smooth_l1_loss(VEL(pred_vertices), VEL(vertices), beta=0.02)
            acc_loss = F.smooth_l1_loss(ACC(pred_vertices), ACC(vertices), beta=0.02)


            geometric_loss = (vertices_loss + (vel_loss + acc_loss)*10) * 1.0  # default weight is 1.0

            joint_bone_vec = (vertices[:,:,:55] - vertices[:, :, SMPLX_PARENTS])[:,:,1:]
            pred_joint_bone_vec = (pred_vertices[:,:,:55] - pred_vertices[:, :, SMPLX_PARENTS])[:,:,1:]

            # bone_vec = joint_bone_vec.reshape((-1, 3))
            # pred_bone_vec = pred_joint_bone_vec.reshape((-1, 3))
            # single_bone_loss = F.cosine_embedding_loss(pred_bone_vec, bone_vec,torch.ones((bone_vec.shape[0],), device=bone_vec.device))
            single_bone_loss = 0

            structure_loss = SimilarityStructureLoss(joint_bone_vec.reshape((-1, 54, 3)), pred_joint_bone_vec.reshape((-1, 54, 3)))


            # perception loss
            # pred_mu, pred_logvar = vae2.E(pred_x1[...,:330 + 3]).chunk(2, dim=-1)
            # gt_mu, gt_logvar = vae2.E(poses_trans[...,:330 + 3]).chunk(2, dim=-1)
            pred_mu, pred_logvar = vae2.encode(pred_x1[...,:330 + 3]).chunk(2, dim=-1)
            gt_mu, gt_logvar = vae2.encode(poses_trans[...,:330 + 3]).chunk(2, dim=-1)
            latent_loss = F.smooth_l1_loss(pred_mu, gt_mu, beta=0.02) * 0.2# default weight is 0.1
            
            # pred_mu, pred_logvar = vae3.E(pred_x1[...,:330 + 3]).chunk(2, dim=-1)
            # gt_mu, gt_logvar = vae3.E(poses_trans[...,:330 + 3]).chunk(2, dim=-1)
            pred_mu, pred_logvar = vae3.encode(pred_x1[...,:330 + 3]).chunk(2, dim=-1)
            gt_mu, gt_logvar = vae3.encode(poses_trans[...,:330 + 3]).chunk(2, dim=-1)
            latent_loss =  latent_loss  + F.smooth_l1_loss(pred_mu, gt_mu, beta=0.02) * 0.2 # default weight is 0.1

            pred_seed_loss = pose_loss_func2(pred_x1[:,:n_seed, :333], poses_trans[:,:n_seed, :333])* 1e-1
            
            
            loss = mse_loss   + geodesic_loss +  geometric_loss + pred_seed_loss + latent_loss + structure_loss + single_bone_loss

            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 1.0)
            optimizer.step()

            if dist.get_rank() == 0:
                batch_loss = batch_loss + loss.item()
                batch_num = batch_num + 1

                if batch_idx % 20 == 0:
                    print(
                        f'EP {epoch} Loss {loss:.6f}  MSE {mse_loss:.6f} LL {latent_loss:.6f} GD {geodesic_loss:.6f} GM {geometric_loss:.6f} VER {vertices_loss:.6f} ST {structure_loss:.6f} SB {single_bone_loss:.6f} Seed {pred_seed_loss:.6f}')
        if dist.get_rank() == 0:
            print(f'Epoch time {(time.time() - epoch_start_time)}')
            writer.add_scalar('train/loss', batch_loss / batch_num, epoch)
            if (epoch + 1) % args.save_n_epoch == 0:
                if not os.path.exists(args.ckpt_folder):
                    # os.mkdir(args.ckpt_folder)
                    os.makedirs(args.ckpt_folder, exist_ok=True)
                torch.save(net.module.state_dict(), os.path.join(
                    args.ckpt_folder, ("epoch_%d") % (epoch,)))


if __name__ == "__main__":
    main()
