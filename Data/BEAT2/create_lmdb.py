
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

"""
This script processes SMPL-X and audio data to create an LMDB database. It includes functions to create the LMDB dataset and test the created dataset by reading and printing data from it.
Functions:
    create_lmdb(data_folder, lmdb_path, seq_size, stride_size):
    create():
        Parses command line arguments to create an LMDB dataset.
            data_folder (str): Path to the smplxflame_30.
            --seq_size (int): Sequence size (default: 60).
            --stride_size (int): Stride size (default: 30).
            --lmdb_path (str): Path to save the LMDB (default: generated based on seq_size and stride_size).
    test():
Usage:
    To create an LMDB dataset:
        python create_lmdb.py <data_folder> --seq_size <seq_size> --stride_size <stride_size> --lmdb_path <lmdb_path>
    To test the created LMDB dataset:
        python create_lmdb.py test <lmdb_path>
"""

import lmdb
import argparse
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import librosa
from scipy.interpolate import interp1d
import sys
sys.path.append("../../Scripts")
from Util import * 
from SMPLX_Util import LocalToGlobalAA
import pandas as pd


    



def DecodeFromBytes(bytes_data, seq_size, betas_shape=300, poses_shape=55*3, expressions_shape=100):
    data = np.frombuffer(bytes_data, dtype=np.float32)
    start_idx = 0
    betas = data[start_idx: start_idx + betas_shape]
    start_idx = start_idx + betas_shape
    poses = data[start_idx: start_idx + poses_shape *
                 seq_size].reshape(seq_size, poses_shape)
    start_idx = start_idx + poses_shape * seq_size
    expressions = data[start_idx: start_idx + expressions_shape *
                       seq_size].reshape(seq_size, expressions_shape)
    start_idx = start_idx + expressions_shape * seq_size
    trans = data[start_idx: start_idx + 3 * seq_size].reshape(seq_size, 3)
    start_idx = start_idx + 3 * seq_size
    sem = data[start_idx: start_idx + 1 * seq_size].reshape(seq_size, 1)
    start_idx = start_idx + 1 * seq_size
    pid = data[start_idx: start_idx + 1]
    start_idx = start_idx + 1
    audio = data[start_idx:]
    return betas, poses, expressions, trans, sem, pid, audio


def LoadSemFile(sem_file: str, wanted_frames):
    with open(sem_file, "r") as fh:
        lines = fh.readlines()

    x = []
    y = []
    for line in lines:
        items = line.strip().split()
        if len(items) >= 5:
            start_time = float(items[1])
            end_time = float(items[2])
            sem_value = float(items[4])
            x.append(start_time + 1e-4)
            y.append(sem_value)
            x.append(end_time - 1e-4)
            y.append(sem_value)
    x = np.array(x)
    y = np.array(y)
    func = interp1d(x, y, assume_sorted=True, fill_value="extrapolate")
    t = np.linspace(0, 1 / 30.0 * wanted_frames, wanted_frames)
    g = func(t)
    return g


def CountData(data_folder):
    smplx_data_folder = os.path.join(data_folder, "smplxflame_30")
    npz_files = glob(os.path.join(smplx_data_folder, "*.npz"))
    num = 0
    for npz_file in npz_files:
        smplx_data = np.load(npz_file, allow_pickle=True)
        poses = smplx_data["poses"].astype(np.float32)
        num += len(poses)
    print("#total frames:", num)


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
    

def create_lmdb(data_folder, lmdb_path, seq_size, stride_size, 
                local_rotation=True, 
                split='train'):
    """
    Processes SMPL-X and audio data to create an LMDB database.

    Args:
        data_folder (str): Path to the folder containing the data.
        lmdb_path (str): Path where the LMDB database will be created.
        seq_size (int): Size of the sequence to be processed.
        stride_size (int): Stride size for processing sequences.

    Returns:
        None
    """
    smplx_data_folder = os.path.join(data_folder, "smplxflame_30")
    audio_data_folder = os.path.join(data_folder, "wave16k")
    sem_data_folder = os.path.join(data_folder, "sem")
    if split =='all':
        npz_files = glob(os.path.join(smplx_data_folder, "*.npz"))
    else:
        split_results = parse_train_test_split_csv(os.path.join(data_folder, "train_test_split.csv"))
        npz_files = [os.path.join(smplx_data_folder, f + ".npz") for f in split_results[split]]

    lmdb_env = lmdb.open(lmdb_path, map_size=2**39)

    if local_rotation == False:
        # pose_mean = GetSMPLXModel().pose_mean.numpy()
        # In flat_hand mode, the pose mean is zero
        pose_mean = 0

    dp_id = 0
    for npz_file in tqdm(npz_files):
        # pid
        pid = os.path.basename(npz_file).split('_')[0]
        pid = np.array(int(pid)).astype(np.float32)

        # audio file
        audio_file = os.path.join(audio_data_folder, os.path.basename(
            npz_file).replace(".npz", ".wav"))
        sem_file = os.path.join(sem_data_folder, os.path.basename(
            npz_file).replace(".npz", ".txt"))
        raw_audio, sr = librosa.load(audio_file, sr=16000)
        raw_audio = raw_audio.astype(np.float32)

        # smplx coefficients
        smplx_data = np.load(npz_file, allow_pickle=True)
        betas = smplx_data["betas"].astype(np.float32)
        poses = smplx_data["poses"].astype(np.float32)
        expressions = smplx_data["expressions"].astype(np.float32)
        trans = smplx_data["trans"].astype(np.float32)

        num_frames = min(len(poses), int(len(raw_audio)/sr * 30))
        num_windows = (num_frames - seq_size) // stride_size + 1
        sem = LoadSemFile(sem_file, num_frames).astype(np.float32)

        poses_lst = [poses[i * stride_size: i * stride_size + seq_size]
                     for i in range(0, num_windows)]
        if local_rotation == False:
            poses_lst = [LocalToGlobalAA(poses_lst[i] + pose_mean).astype(np.float32) for i in range(len(poses_lst))]
        expressions_lst = [expressions[i * stride_size: i *
                                       stride_size + seq_size] for i in range(0, num_windows)]
        trans_lst = [trans[i * stride_size: i * stride_size + seq_size]
                     for i in range(0, num_windows)]
        sem_lst = [sem[i * stride_size: i * stride_size + seq_size]
                   for i in range(0, num_windows)]
        # audio_lst = [
        #     raw_audio[(i * stride_size) * sr // 30: (i * stride_size + seq_size) * sr // 30] for i in range(0, num_windows)
        # ]
        audio_stride = seq_size * sr // 30
        audio_lst = [
            raw_audio[(i * stride_size) * sr // 30: (i * stride_size ) * sr // 30 + audio_stride] for i in range(0, num_windows)
        ]
        with lmdb_env.begin(write=True) as txn:
            for i in range(num_windows):
                assert len(poses_lst[i]) == seq_size
                assert len(expressions_lst[i]) == seq_size
                assert len(trans_lst[i]) == seq_size
                assert len(sem_lst[i]) == seq_size
                # assert len(audio_lst[i]) == sr * seq_size // 30
                bytes_data = (
                    betas.tobytes()
                    + poses_lst[i].tobytes()
                    + expressions_lst[i].tobytes()
                    + trans_lst[i].tobytes()
                    + sem_lst[i].tobytes()
                    + pid.tobytes()
                    + audio_lst[i].tobytes()
                )
                
                txn.put(f"{dp_id:010d}".encode("utf-8"), bytes_data)
                txn.put(f"{dp_id:010d}f".encode("utf-8"), os.path.basename(npz_file).encode("utf-8"))
                dp_id = dp_id + 1

    with lmdb_env.begin(write=True) as txn:
        print("#num", dp_id)
        txn.put("num".encode("utf-8"), str(dp_id).encode("utf-8"))
        txn.put("betas_shape".encode("utf-8"), str(300).encode("utf-8"))
        txn.put("poses_shape".encode("utf-8"), str(165).encode("utf-8"))
        txn.put("expressions_shape".encode("utf-8"), str(100).encode("utf-8"))
        txn.put("trans_shape".encode("utf-8"), str(3).encode("utf-8"))
        txn.put("sem_shape".encode("utf-8"), str(1).encode("utf-8"))
        txn.put("seq_size".encode("utf-8"), str(seq_size).encode("utf-8"))


def create():
    parser = argparse.ArgumentParser(description="Create LMDB dataset")
    parser.add_argument("data_folder", type=str,
                        help="Path to the smplxflame_30")
    parser.add_argument("--seq_size", type=int, default="60",
                        help="Sequence size in frames")
    parser.add_argument("--stride_size", type=int,
                        default="20", help="Stride size in frames")
    parser.add_argument("--lmdb_path", type=str, default="",
                        help="Path to save the lmdb")
    parser.add_argument("--local_rotation", type=int, default="0",
                        help="Path to save the lmdb")
    parser.add_argument("--split", type=str, default="train", choices=['all', 'train', 'test', 'val', 'additional'], help="(train, val, test, additional)")
    args = parser.parse_args()

    if args.lmdb_path == "":
        local_global = "local" if args.local_rotation!=0 else "global"
        args.lmdb_path = f"{args.split}_seq_size_{args.seq_size}_stride_size_{args.stride_size}_{local_global}.lmdb"

    create_lmdb(
        data_folder=args.data_folder, lmdb_path=args.lmdb_path, seq_size=args.seq_size, 
        stride_size=args.stride_size,
        local_rotation=args.local_rotation!=0,
        split=args.split
    )


def test():
    import soundfile as sf
    """
    Test function to read and print data from an LMDB dataset.
    This function parses command line arguments to get the path to the LMDB dataset,
    opens the LMDB environment, and reads various data attributes and values from it.
    It prints the shapes of the data attributes and values read from the dataset.
    Command line arguments:
    lmdb_path (str): Path to save the LMDB dataset.
    The function reads the following data attributes from the LMDB dataset:
    - num: Number of data points.
    - betas_shape: Shape of the betas data.
    - poses_shape: Shape of the poses data.
    - expressions_shape: Shape of the expressions data.
    - seq_size: Sequence size.
    The function reads the following data values from the LMDB dataset:
    - betas: Betas data.
    - poses: Poses data.
    - expressions: Expressions data.
    - trans: Translation data.
    - audio: Audio data.
    The shapes of the data values are printed to the console.
    """
    parser = argparse.ArgumentParser(description="Create LMDB dataset")
    parser.add_argument("lmdb_path", type=str, default="",
                        help="Path to save the lmdb")
    args = parser.parse_args()

    lmdb_env = lmdb.open(args.lmdb_path)
    with lmdb_env.begin() as txn:
        num = int(txn.get("num".encode("utf-8")).decode("utf-8"))
        betas_shape = int(
            txn.get("betas_shape".encode("utf-8")).decode("utf-8"))
        poses_shape = int(
            txn.get("poses_shape".encode("utf-8")).decode("utf-8"))
        expressions_shape = int(
            txn.get("expressions_shape".encode("utf-8")).decode("utf-8"))
        sem_shape = int(txn.get("sem_shape".encode("utf-8")).decode("utf-8"))
        seq_size = int(txn.get("seq_size".encode("utf-8")).decode("utf-8"))

        print(
            f"num {num} betas_shape {betas_shape} poses_shape {poses_shape} expressions_shape {expressions_shape} seq_size {seq_size}"
        )

        dp_id = 1000
        bytes_data = txn.get(f"{dp_id:010d}".encode("utf-8"))
        betas, poses, expressions, trans, sem, pid, audio = DecodeFromBytes(
            bytes_data=bytes_data, seq_size=seq_size, betas_shape=betas_shape, poses_shape=poses_shape, expressions_shape=expressions_shape)
        print(pid)
        print(
            f"betas {betas.shape} poses {poses.shape} expressions {expressions.shape} trans {trans.shape} sem {sem.shape} audio {audio.shape}"
        )
        np.savez(
            "test.npz",
            betas=betas,
            poses=poses,
            expressions=expressions,
            trans=trans,
            gender=np.array("neutral"),
            mocap_frame_rate=np.array(30),
            model="smplx2020",
        )

        sf.write("test.wav", audio.reshape(-1), samplerate=16000)


if __name__ == "__main__":
    create()
