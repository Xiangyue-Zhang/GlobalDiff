
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
sys.path.append("..")
import torch
import torch.utils.data.distributed
import torch.distributed as dist
import random
import numpy as np
from VAE import MaskedVAE3, ReconsLoss
from BEATXDataset import LMDBDataset as BEATXDataset
from torch.optim.lr_scheduler import StepLR
import time
from einops import rearrange
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from SMPLX_Util import GetSMPLXModel, SplitPosesIntoSmplxParams
from Util import rotation_6d_to_axis_angle, rotation_6d_to_matrix, axis_angle_to_rotation_6d
from FK import GFKWraper



class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, beat, motionx):
        super().__init__()
        self.beat = beat
        self.motionx = motionx
    
    def __len__(self):
        return len(self.beat) + len(self.motionx)
    
    def __getitem__(self, index):
        
        if index < len(self.beat):
            data =  self.beat[index]
        else:
            data =  self.motionx[index - len(self.beat)]
        
        return {
            "poses": data["poses"],
            "trans": data["trans"]
        }


def VEL(x: torch.Tensor):
    return x[:, 1:] - x[:, :-1]


def ACC(x: torch.Tensor):
    return VEL(VEL(x))

def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # 这一句话会使VQVAE的encoder变的很慢

def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return torch.mean(kl)  # 直接对所有维度求平均

def kl_weight(current_epoch, warp_up_epoch=10, kl_weight=1.0):
    if warp_up_epoch <= 0:
        return kl_weight
    else:
        return min(current_epoch / warp_up_epoch, 1.0) * kl_weight  # 前 10 个 epoch 逐渐增加



def GetSmplxVerts(x, smplx_model, betas_input=None):
        """
        x:B,T,55 * 3
        """
        B, T, _ = x.shape
        smplx_params = SplitPosesIntoSmplxParams(x)
        for k in smplx_params.keys():
            smplx_params[k] = smplx_params[k].reshape((B * T, -1))
        # betas = betas.reshape((B, 1, -1)).repeat((1, T, 1)).reshape((B * T, -1))
        betas = torch.zeros((B*T, 300), device=x.device) if betas_input is None else betas_input.reshape((B*T, 300))
        output = smplx_model(betas=betas,
                                  global_orient=smplx_params["global_orient"],
                                  body_pose=smplx_params["body_pose"],
                                  expression=torch.zeros((B * T, 100), device=x.device),
                                  jaw_pose=smplx_params["jaw_pose"],
                                  leye_pose=smplx_params["leye_pose"],
                                  reye_pose=smplx_params["reye_pose"],
                                  left_hand_pose=smplx_params["left_hand_pose"],
                                  right_hand_pose=smplx_params["right_hand_pose"],
                                  transl=torch.zeros((B * T, 3 ), device=x.device),
                                  return_verts=True)
        vertices = output.vertices.reshape((B, T, -1))
        return vertices



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str,
                        default="../../Data/BEAT2/train_seq_size_60_stride_size_20_global.lmdb")
    parser.add_argument("--test_lmdb_path", type=str,
                        default="../../Data/BEAT2/test_seq_size_60_stride_size_20_global.lmdb")
    parser.add_argument("--ckpt_folder", type=str, default="ckpt/split/global/MaskedVAE3-HorizonFlip-alpha80")
    parser.add_argument("--batch_size", type=int, default="128")
    parser.add_argument("--num_workers", type=int, default="4")
    parser.add_argument("--epoch", type=int, default="200")
    parser.add_argument("--lr", type=float, default="0.0002")
    parser.add_argument("--local_rank", type=int,
                        help="local rank, will passed by ddp")
    parser.add_argument("--local-rank", type=int,
                        help="local rank, will passed by ddp")
    parser.add_argument("--sched_step", type=int, default="1000")
    parser.add_argument("--save_n_epoch", type=int, default="10")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default="0")
    parser.add_argument("--kld", type=float, default="1e-4")
    parser.add_argument("--include_translation", type=int, default="1")
    parser.add_argument("--fp16", type=int, default="0")

    args = parser.parse_args()
    setup_seed(args.seed)

    if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] is not None:
        print(os.environ["LOCAL_RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl", rank=args.local_rank, init_method="env://")

    train_dataset = BEATXDataset(lmdb_path=args.lmdb_path, use_symmetry_aug=True, rotation_method='axisangle')
    test_dataset = BEATXDataset(lmdb_path=args.test_lmdb_path)

    GFK = GFKWraper().cuda()
    if dist.get_rank() == 0:
        print('#train_dataset', len(train_dataset),
              '#test_dataset', len(test_dataset))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, drop_last=True
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    net = MaskedVAE3(in_channels=55 * 6 + (3 if args.include_translation else 0), 
               hidden_size=256,
            #    latent_dim=128,
               pred_logvar=True).cuda()
    if args.resume != "":
        ckpt = torch.load(args.resume)
        net.load_state_dict(ckpt)

    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank], find_unused_parameters=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    best_test_rec = 9999.0
    loss_func = ReconsLoss(
        55 * 6, 3 if args.include_translation else 0, rotation_method='rot6d')
    # loss_func = F.l1_loss

    ckpt_folder = args.ckpt_folder
    
    if dist.get_rank() == 0:
        log_dir = os.path.join(ckpt_folder, "log")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(args.epoch):
        net.train()
        if dist.get_rank() == 0:
            epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)
        batch_loss = 0
        batch_num = 0
        for batch_idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            poses = axis_angle_to_rotation_6d(data['poses'].to(args.local_rank))
            trans = data['trans'].to(args.local_rank)
            rest_joints = data['J'].cuda().unsqueeze(1).expand((-1, poses.shape[1], -1, -1)).reshape((-1, 55,  3))
            B, L = poses.shape[:2]
            # shape_betas = betas.reshape((len(betas), 1, 300)).repeat((1, poses.shape[1], 1))
            # vertices = GetSmplxVerts(ConvertRot6dToAxisAngle(poses), smplx_model, shape_betas)
            # poses = vertices
            if args.include_translation:
                motion = torch.cat([poses, trans], dim=2)
            else:
                motion = poses
            
            mask = torch.rand((B, L)) < 0.8 # default 0.3
            rec, mu, logvar = net(motion, mask)
            logvar = torch.clamp(logvar, min=-10, max=10)
            KLD = kl_loss(mu, logvar) * kl_weight(epoch, warp_up_epoch=10, kl_weight=args.kld)


            rec_loss = loss_func(motion, rec)
            vel_loss = loss_func.forward_vel(rec, motion) * 0.05
            smooth_loss = loss_func.forward_smooth(rec, motion) * 0.05
            
            
            
            gt_posemat = rotation_6d_to_matrix(poses[...,:330].reshape((-1, 55, 6)))
            pred_posemat  = rotation_6d_to_matrix(rec[...,:330].reshape((-1, 55, 6)))
            vertices = GFK(pred_posemat, rest_joints).reshape((B, L, -1, 3))
            pred_vertices = GFK(gt_posemat, rest_joints).reshape((B, L, -1, 3)) 
            if args.include_translation:
                pred_trans = rec[..., 330:333]
                vertices = vertices + trans.unsqueeze(-2)
                pred_vertices = pred_vertices + pred_trans.unsqueeze(-2)
            G_vertices_loss = F.smooth_l1_loss(pred_vertices, vertices, beta=0.02)
            G_vel_loss = F.smooth_l1_loss(VEL(pred_vertices), VEL(vertices), beta=0.02)
            G_acc_loss = F.smooth_l1_loss(ACC(pred_vertices), ACC(vertices), beta=0.02)
            geometric_loss = G_vertices_loss + G_vel_loss + G_acc_loss


            loss = rec_loss + KLD + vel_loss + smooth_loss +  geometric_loss
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 1.0) 
            optimizer.step()

            if dist.get_rank() == 0:
                batch_loss = batch_loss + loss.item()
                batch_num = batch_num + 1
                if (batch_idx + 1) % 20 == 0:
                    print(
                        f'Epoch {epoch} Loss {loss:0.6f} REC {rec_loss:0.6f} GM {geometric_loss:0.6f} VERT {G_vertices_loss:0.6f} VEL {G_vel_loss:0.6f} KLD {KLD:0.6f}')
                    
        if dist.get_rank() == 0:
            writer.add_scalar('train/REC', batch_loss / batch_num, epoch)
            print(
                f'Epoch {epoch} Epoch time {(time.time() - epoch_start_time)}')
            if (epoch + 1) % args.save_n_epoch == 0:
                if not os.path.exists(ckpt_folder):
                    os.makedirs(ckpt_folder, exist_ok=True)
                torch.save(net.module.state_dict(), os.path.join(
                    ckpt_folder, ("epoch_%d") % (epoch,)))

            # test
            if (epoch + 1) % 5 == 0:
                net.eval()
                rec_score = 0.0
                n_test_batch = 0
                for batch_idx, data in enumerate(test_dataloader):
                    poses = data['poses'].to(args.local_rank)
                    trans = data['trans'].to(args.local_rank)
                    if args.include_translation:
                        motion = torch.cat([poses, trans], dim=2)
                    else:
                        motion = poses
                    with torch.no_grad():
                        rec, mu, logvar = net(motion,reparameterize=False)
                        rec_score = rec_score + loss_func(motion, rec)
                        n_test_batch = n_test_batch + 1
                if best_test_rec > rec_score / n_test_batch:
                    best_test_rec = rec_score / n_test_batch
                    if not os.path.exists(ckpt_folder):
                        os.makedirs(ckpt_folder, exist_ok=True)
                    torch.save(net.module.state_dict(), os.path.join(
                        ckpt_folder, "best.pt"))
                print(
                    f'---------------EPOCH {epoch} TEST REC {rec_score/n_test_batch: 0.6f} BEST REC {best_test_rec:0.6f}---------------')
                writer.add_scalar('train/BEST_REC', best_test_rec.detach(), epoch)


if __name__ == "__main__":
    main()
