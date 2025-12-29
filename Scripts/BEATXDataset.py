

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

import os
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from Util import axis_angle_to_rotation_6d
from  scipy.io import loadmat
from SMPLX_Util import SMPLX_PARENTS, symmetrize_smplx_parameters
import random


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, 
                 lmdb_hubert=None, 
                 audio_preprocessor=None, 
                 body_part_indices=None,
                 rotation_method="rot6d",
                 use_symmetry_aug=False):
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        txn = self.env.begin(write=False) 
        self.num = int(txn.get("num".encode("utf-8")).decode("utf-8"))
        self.betas_shape = int(txn.get("betas_shape".encode("utf-8")).decode("utf-8"))
        self.poses_shape = int(txn.get("poses_shape".encode("utf-8")).decode("utf-8"))
        self.expressions_shape = int(txn.get("expressions_shape".encode("utf-8")).decode("utf-8"))
        self.seq_size = int(txn.get("seq_size".encode("utf-8")).decode("utf-8"))
        self.txn = txn

        self.env_hubert = None
        if lmdb_hubert is not None:
            self.env_hubert = lmdb.open(lmdb_hubert, readonly=True, lock=False)
            self.txn_hubert = self.env_hubert.begin(write=False) 
        
        self.audio_preprocessor = audio_preprocessor
        self.body_part_indices = body_part_indices
        self.rotation_method = rotation_method
        assert rotation_method =="rot6d" or rotation_method =="axisangle"

        self.rest_joint_position = {}
        for k, v in loadmat(os.path.join(os.path.dirname(__file__),'rest_joint_position.mat')).items():
            if k.isdigit():
                self.rest_joint_position[int(k)] = torch.from_numpy(v).float()
        self.bone_length = {}
        for k in self.rest_joint_position:
            v = self.rest_joint_position[k]
            bone_length = v - v[SMPLX_PARENTS]
            bone_length = torch.norm(bone_length, dim=-1)
            self.bone_length [k] = bone_length[1:]
        self.use_symmetry_aug = use_symmetry_aug 
        if use_symmetry_aug:
            assert rotation_method == "axisangle", "symmetry augmentation only support axisangle"
                
    def __len__(self):
        return self.num

    def evalute_mean_std(self, progressive=False):
        mu = 0
        loop_range = range(self.num)
        if progressive:
            from tqdm import tqdm
            loop_range = tqdm(loop_range, total=self.num)
        for i in loop_range:
            mu += self.__getitem__(i)["poses"]
        mu = mu / self.num

        std = 0
        loop_range = range(self.num)
        if progressive:
            from tqdm import tqdm
            loop_range = tqdm(loop_range, total=self.num)
        for i in loop_range:
            diff = mu - self.__getitem__(i)["poses"]
            std += diff ** 2
        std = std / self.num
        std = torch.sqrt(std)
        return mu, std

    def _get_data_item(self, idx):
        # with self.env.begin(write=False) as txn:
        txn = self.txn
        dp_id = idx
        bytes_data = txn.get(f"{dp_id:010d}".encode("utf-8"))
        data = np.frombuffer(bytes_data, dtype=np.float32)
        start_idx = 0
        betas = data[start_idx : start_idx + self.betas_shape]
        start_idx = start_idx + self.betas_shape
        poses = data[start_idx : start_idx + self.poses_shape * self.seq_size].reshape(self.seq_size, self.poses_shape)
        start_idx = start_idx + self.poses_shape * self.seq_size
        expressions = data[start_idx : start_idx + self.expressions_shape * self.seq_size].reshape(self.seq_size, self.expressions_shape)
        start_idx = start_idx + self.expressions_shape * self.seq_size
        trans = data[start_idx : start_idx + 3 * self.seq_size].reshape(self.seq_size, 3)
        start_idx = start_idx + 3 * self.seq_size
        sem = data[start_idx : start_idx + 1 * self.seq_size].reshape(self.seq_size, 1)
        start_idx = start_idx + 1 * self.seq_size
        pid = data[start_idx : start_idx + 1]
        start_idx = start_idx + 1
        audio = data[start_idx:]
            
            # bytes_data = txn.get(f"{dp_id:010d}f".encode("utf-8"))
            # npz_file = str(bytes_data,encoding="utf-8")
            # print(npz_file)
        return betas, poses, expressions,trans, sem, pid, audio
    def __getitem__(self, idx):
        betas, poses_aa, expressions,trans, sem, pid, audio = self._get_data_item(idx)
        if self.body_part_indices is not None:
            poses_aa = poses_aa[:, self.body_part_indices]
        
        if self.rotation_method == "rot6d":
            poses = axis_angle_to_rotation_6d(poses_aa).astype(np.float32)
        elif self.rotation_method == "axisangle":
            poses = poses_aa
        
        betas = torch.from_numpy(betas.copy())
        poses = torch.from_numpy(poses.copy())
        expressions = torch.from_numpy(expressions.copy())
        trans = torch.from_numpy(trans.copy())
        sem = torch.from_numpy(sem.copy())
        audio = audio.copy()
        
        assert len(audio) == 16000 * len(poses)//30
        
        if self.audio_preprocessor is not None:
            audio = self.audio_preprocessor(audio)
        audio = torch.from_numpy(audio)
        
        if self.env_hubert is not None:
            # with self.env_hubert.begin(write=False) as txn:
            txn = self.txn_hubert
            bytes_data = txn.get(f"{idx:010d}".encode("utf-8"))
            hubert = np.frombuffer(bytes_data, dtype=np.float32).copy().reshape((self.seq_size, 1024))
            hubert = torch.from_numpy(hubert)
        else:
            hubert = 0 
            
        J = self.rest_joint_position[int(pid[0])]
        pid_offset = 0
        if self.use_symmetry_aug and random.random() < 0.5:
            poses, trans = symmetrize_smplx_parameters(poses, trans)
            pid_offset = 30
            J[..., 0] = -J[..., 0]
            
        return {
            "betas": betas,
            "poses": poses,
            "expressions": expressions,
            "trans":trans,
            "sem":sem,
            "audio":audio,
            "hubert":hubert,
            "pid":int(pid[0]) + pid_offset,
            "J": J,
            "bone_length":self.bone_length[int(pid[0])],
        }







# Example usage
if __name__ == "__main__":
    
    import soundfile as sf
    from Util import *
    from SMPLXConfig import GetBodyPartIndices
    from SMPLX_Util import GlobalToLocalAA, symmetrize_smplx_parameters
    from scipy.spatial.transform import Rotation as R
    # lmdb_path = "/mnt/4TDisk/mm_data/mm/data/BEAT/V2/seq_size_240_stride_size_60.lmdb"
    lmdb_path = "/mnt/4TDisk/mm_data/mm/data/BEAT/V2/seq_size_60_stride_size_20_global.lmdb"
    # lmdb_hubert = "/mnt/4TDisk/mm_data/mm/data/BEAT/V2/seq_size_240_stride_size_60._hubert.lmdb"
    lmdb_path = "/mnt/4TDisk/mm_data/mm/project/LargeMotionModel/Data/BEAT2/train_seq_size_60_stride_size_20_local.lmdb"
    lmdb_hubert = None
    ds = LMDBDataset(lmdb_path, lmdb_hubert, rotation_method="axisangle")


    data_id = 4569
    betas = ds[data_id]["betas"]
    poses = ds[data_id]["poses"]
    sem = ds[data_id]["sem"].numpy()
    audio = ds[data_id]["audio"].numpy()
    expressions = ds[data_id]["expressions"].numpy()
    trans = ds[data_id]["trans"].numpy()
    SMPL_X_FLIP_PAIRS = ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), # body
                     (23,24), #eyes
                     (25,40), (26,41), (27,42), (28,43), (29,44), (30,45), (31,46), (32,47), (33,48), (34,49), (35,50), (36,51), (37,52), (38,53), (39,54) # hands
                    )
    sym_poses, sym_trans = symmetrize_smplx_parameters(poses, torch.from_numpy(trans))



    np.savez(
            "test.npz",
            betas=betas,
            poses=sym_poses,
            expressions=expressions,
            trans=sym_trans,
            gender=np.array("neutral"),
            mocap_frame_rate=np.array(30),
            model="smplx2020",
    )
    print("#pid", ds[data_id]["pid"])
    sf.write("test.wav", audio, samplerate=16000)