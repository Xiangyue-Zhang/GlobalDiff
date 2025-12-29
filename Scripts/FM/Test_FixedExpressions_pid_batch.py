
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
from transformers import Wav2Vec2Processor, HubertModel, AutoModelForCTC,WavLMModel
from Util import *
from VAE.VAE import VAE
from einops import rearrange
import random
from Util import  NormalizeRot6d, rotation_6d_to_matrix, slerp_torch, rotation_matrix_to_quaternion_torch, quaternion_to_matrix_torch, matrix_to_rotation_6d
# from Net import  DiffusionDITNetParts as MotionModel
from Net import DiffusionDITNetPartsFixedExpressions2PostNorm as MotionModel
from SMPLX_Util import GetSMPLXModel, SplitPosesIntoSmplxParams, global_to_local_batch, LocalToGlobalAA
from Util import rotation_6d_to_axis_angle
from SMPLXConfig import GetBodyPartIndices
import pandas as pd
import os
from tqdm import tqdm
from Net import SimpleSpeechModel
from HFModelConfig import MODEL_NAMES

def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def linear_interpolate(x0, x1, t):
    """
    线性插值函数
    :param x0: 起始点
    :param x1: 结束点
    :param t: 插值参数，范围在[0, 1]
    :return: 插值结果
    """
    return (1 - t) * x0 + t * x1


class AudioFeatureExtractor(nn.Module):
    def __init__(self, model_id=MODEL_NAMES["wavlm"]):
        super(AudioFeatureExtractor, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = WavLMModel.from_pretrained(model_id)

    def extract(self, audio, overlap_frame_size=8):
        if isinstance(audio, str):
            raw_audio, sr = librosa.load(audio, sr=16000)
        else:
            raw_audio = audio
            sr = 16000
        raw_audio_clips = SplitIntoOverlapedWindows(raw_audio,win_size=sr * 2, stride_step=sr * 2 - int(sr / 30 * overlap_frame_size), keep_last=False)
    
        huberts = []
        for aud_idx, aud_clip in enumerate(raw_audio_clips):
            input_values = self.processor(aud_clip, return_tensors='pt', sampling_rate=16000).input_values.cuda()
            with torch.no_grad():
                audio_embeding = self.model.forward(input_values).last_hidden_state
                audio_embeding = F.interpolate(audio_embeding.permute((0, 2, 1)), 60, mode='linear', align_corners=True).permute((0, 2, 1))
            huberts.append(audio_embeding)
        return huberts


def parse_train_test_split_csv(csv_file):
    tables = pd.read_csv(csv_file)
    file_ids = tables['id']
    file_types = tables['type']
    train_files = []
    test_files =[]
    val_files = []
    additional_files = []
    for n in range(len(file_ids)):
        if file_types[n] == 'train':
            train_files.append(file_ids[n])
        elif file_types[n] == 'test':
            test_files.append(file_ids[n])
        elif file_types[n] == 'val':
            val_files.append(file_ids[n])
        elif file_types[n] == 'additional':
            additional_files.append(file_ids[n])
        else:
            assert 0 == 1
    return {
        "train":train_files,
        "test":test_files,
        "val":val_files,
        "traadditionalin":additional_files,
    }
    

def check_best(recored_file):

    valid_lines = []
    with open(recored_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() != "":
            valid_lines.append(float(line.strip()))


    if len(valid_lines) == 1:
        return True
    else:
        for i in range(0, len(valid_lines)-1):
            if valid_lines[-1] > valid_lines[i]:
                return False
        return True




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_folder", type=str)
    parser.add_argument("--outdir", type=str, default="tmp_test_out")
    parser.add_argument("--pid", type=str, default="2")
    parser.add_argument("--data_root", type=str, default="/mnt/4TDisk/mm_data/mm/data/BEAT/V2/beat_v2.0.0/beat_english_v2.0.0/")
    parser.add_argument("--seed", type=int, default="0")
    parser.add_argument("--local", type=int, default="0")
    parser.add_argument('--wav2vec', type=str, default=MODEL_NAMES["wavlm"])
    args = parser.parse_args()

    setup_seed(0)
    
    
    hand_indices = GetBodyPartIndices("hand", "rot6d")
    upper_indices = GetBodyPartIndices("upper", "rot6d")
    lower_indices = GetBodyPartIndices("lower", "rot6d")
    lhand_indices = GetBodyPartIndices("lhand", "rot6d")
    rhand_indices = GetBodyPartIndices("rhand", "rot6d")
    trans_indices = np.arange(55 * 6, 55 *6 + 3)
    lower_indices = np.concatenate([lower_indices, trans_indices], axis=0)
    face_joint_indices = GetBodyPartIndices("face", "rot6d")
    expression_indices = np.arange(55 * 6 + 3, 55 *6 + 3 + 100)
    face_indices = np.concatenate([face_joint_indices, expression_indices], axis=0)
    
    
    n_seed = 8

    simple_speech_model = SimpleSpeechModel().cuda().requires_grad_(False).eval()
    # simple_speech_model.load_state_dict(torch.load("ckpt/split/SimpleSpeechModel-GeometricLoss/best.pt"))
    simple_speech_model.load_state_dict(torch.load("ckpt/split/SimpleSpeechModel/best.pt"))


    net = MotionModel( n_seed=n_seed, num_person=31).cuda().requires_grad_(False).eval()
    


    test_file_names = parse_train_test_split_csv(os.path.join(args.data_root, "train_test_split.csv"))['test']
    if args.pid != "all":
        test_file_names = [name for name in test_file_names if name.split("_")[0] == args.pid]
    
    audio_files = [os.path.join(args.data_root, "wave16k", name + ".wav") for name in test_file_names]
    npz_files = [os.path.join(args.data_root, "smplxflame_30", name + ".npz") for name in test_file_names]


    

    audio_npy = [librosa.load(audio, sr=16000)[0] for audio in audio_files]
    max_len = max([len(audio) for audio in audio_npy])
    padded_audio = [np.pad(audio, (0, max_len - len(audio)), 'constant') for audio in audio_npy]
    
    audio_feature_extractor = AudioFeatureExtractor().requires_grad_(False).cuda().eval()
    huberts = []
    for audio in tqdm(padded_audio, total=len(padded_audio)):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            h = audio_feature_extractor.extract(audio)
            h = torch.cat(h, dim=0)
            huberts.append(h)
    
    huberts = torch.stack(huberts, dim=0)

    
    beta_list = []
    pose_list = []
    pid_list = []
    for npz_file in npz_files:
        smplx_data = np.load(npz_file, allow_pickle=True)
        poses = smplx_data["poses"].astype(np.float32)[:8]
        if args.local == 0:
            poses = LocalToGlobalAA(poses)
        poses = axis_angle_to_rotation_6d(poses)
        trans = smplx_data["trans"].astype(np.float32)[:8]
        pose_list.append(np.concatenate([poses, trans], axis=-1))
        betas = smplx_data["betas"].astype(np.float32)
        beta_list.append(betas)
        pid = torch.LongTensor([int(npz_file.split("/")[-1].split("_")[0])])
        pid_list.append(pid)
    

    pose_list = torch.from_numpy(np.stack(pose_list, axis=0)).cuda().float()
    pid_list = torch.cat(pid_list, dim=0).cuda()
    num_segments = huberts.shape[1]
    B = huberts.shape[0]
    out_file = os.path.join(args.ckpt_folder, f"record_{args.pid}.txt")
    if os.path.exists(out_file):
        os.remove(out_file)
    from Filter import Filter1D
    smoothing_window_length = 7
    filter = Filter1D(window_length=smoothing_window_length, polyorder=2).cuda()
    for epoch in tqdm(range(19,999 + 1, 20), total=50):
        net.load_state_dict(torch.load(os.path.join(args.ckpt_folder, f"epoch_{epoch}")))

        seed_poses = pose_list[:,:n_seed]
        motions = []
        for n in  range(num_segments):
            pred_expressions = simple_speech_model(huberts[:,n], pid_list)
            model_kwargs = {
                    "pid": pid_list,
                    "seed_poses": seed_poses,
                    "hand_indices": hand_indices,
                    "upper_indices": upper_indices,
                    "lower_indices": lower_indices,
                    "face_indices": face_indices,
                    "face_joint_indices": face_joint_indices,
                    "hubert":huberts[:,n],
                    "expressions":pred_expressions,
                    "lhand_indices": lhand_indices,
                    "rhand_indices": rhand_indices,
                    }
            cfg_model_kwargs = {
                    "pid": torch.zeros_like(pid_list),
                    "seed_poses": seed_poses,
                    "hand_indices": hand_indices,
                    "upper_indices": upper_indices,
                    "lower_indices": lower_indices,
                    "face_indices": face_indices,
                    "face_joint_indices": face_joint_indices,
                    "hubert":huberts[:,n],
                    "expressions":pred_expressions,
                    "lhand_indices": lhand_indices,
                    "rhand_indices": rhand_indices,
                    }
            x0 = torch.randn((B, 60, 55 * 6 + 3)).cuda()
            xt = x0
            n_steps = 5
            cfg_scale = 0
            timesteps = torch.linspace(0, 1, n_steps + 1).cuda()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                for i in range(n_steps):
                    ts, te = timesteps[i], timesteps[i + 1]
                    h = (te - ts) / 2.0
                    pred_x1 = net(xt, ts.expand(B), **model_kwargs)
                    # pred_x1[...,:330] = NormalizeRot6d(pred_x1[...,:330])
                    if cfg_scale > 0:
                        pred_x1_cfg = net(xt, ts, **cfg_model_kwargs)
                        pred_x1 = (pred_x1 - pred_x1_cfg) * cfg_scale + pred_x1_cfg
                    
                    pred_x1[:,:n_seed] = (model_kwargs["seed_poses"] + pred_x1[:,:n_seed]) / 2.0
                    xt = linear_interpolate(x0, pred_x1, ts + h)
                    pred_x1 = net(xt, (ts+h).expand(B), **model_kwargs)
                    # pred_x1[...,:330] = NormalizeRot6d(pred_x1[...,:330])
                    if cfg_scale > 0:
                        pred_x1_cfg = net(xt, ts+h, **cfg_model_kwargs)
                        pred_x1 = (pred_x1 - pred_x1_cfg) * cfg_scale + pred_x1_cfg
                    
                    pred_x1[:,:n_seed] = (model_kwargs["seed_poses"] + pred_x1[:,:n_seed]) / 2.0
                    # xt = (pred_x1 - xt) / (1 - ts - h)*(te - 1) + pred_x1
                    xt = linear_interpolate(x0, pred_x1, te)

            xt[:,n_seed-smoothing_window_length//2:]  = filter(xt[:,n_seed-smoothing_window_length//2:])
            xt[...,:330] = NormalizeRot6d(xt[...,:330])
            seed_poses = xt[:,-n_seed:][..., :333]
            xt = torch.cat([xt, pred_expressions], dim=-1)
            motions.append(xt)
        for i in range(1, len(motions)):
            motions[i-1][:,-n_seed:] = (motions[i-1][:,-n_seed:] + motions[i][:,:n_seed])/2.0
            motions[i] = motions[i][:,n_seed:]
        
        sample_rot6d = torch.cat(motions, dim=1)
        expressions = sample_rot6d[...,330 + 3:]
        if args.local == 0:
            poses = rotation_6d_to_matrix(sample_rot6d[...,:330].reshape(-1, 55, 6))
            poses = global_to_local_batch(poses)
            poses = rotation_matrix_to_axis_angle(poses).reshape(B, -1, 55 * 3)
        else:
            poses = sample_rot6d[...,:330].reshape(B, -1, 55 * 6)
            poses = rotation_6d_to_axis_angle(poses)
        pred_trans = sample_rot6d[...,330:330 + 3:]


        sr = 16000
        win_size = sr * 2
        stride_step = sr * 2 - int(sr / 30 * n_seed)
        pred_poses_list = []
        pred_trans_list = []
        pred_expressions_list = []
        for i in range(len(audio_files)):
            n = (len(audio_npy[i]) - (win_size - stride_step)) // stride_step
            assert n >= 2
            n_frames = 60 + (60 - n_seed) * (n-1)
            pred_poses_list.append(poses[i,:n_frames].cpu().numpy())
            pred_trans_list.append(pred_trans[i,:n_frames].cpu().numpy())
            pred_expressions_list.append(expressions[i,:n_frames].cpu().numpy())
            
        
        id = 0
        for npz_file, betas in tqdm(zip(npz_files, beta_list), total=len(npz_files)):
            os.makedirs(args.outdir, exist_ok=True)
            np.savez(
                os.path.join(args.outdir, "res_" + os.path.basename(npz_file)),
                betas=betas,
                poses=pred_poses_list[id],
                expressions=pred_expressions_list[id],
                trans=pred_trans_list[id],
                gender=np.array("neutral"),
                mocap_frame_rate=np.array(30),
                model="smplx2020",
            )
            id += 1
            # return
        

        cmd = "python Test_FID_From_Results.py " + " --outfile=" + out_file + ' --data_folder=' + args.outdir
        ret = os.system(cmd)
        if ret !=0:
            return

        if check_best(out_file):
            id = 0
            best_outdir = os.path.join(args.ckpt_folder, f"best_pid_{args.pid}")
            for npz_file, betas in tqdm(zip(npz_files, beta_list), total=len(npz_files)):
                os.makedirs(best_outdir, exist_ok=True)
                np.savez(
                    os.path.join(best_outdir, "res_" + os.path.basename(npz_file)),
                    betas=betas,
                    poses=pred_poses_list[id],
                    expressions=pred_expressions_list[id],
                    trans=pred_trans_list[id],
                    gender=np.array("neutral"),
                    mocap_frame_rate=np.array(30),
                    model="smplx2020",
                )
                id += 1
    

if __name__ == '__main__':
    main()
