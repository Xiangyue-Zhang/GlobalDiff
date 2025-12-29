
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
from transformers import Wav2Vec2Processor, WavLMModel
import torch.nn.functional as F
import torch
from tqdm import tqdm
from create_lmdb import DecodeFromBytes
from HFModelConfig import MODEL_NAMES

def create():
    """
    Creates an LMDB dataset with audio embeddings using the HuBERT model.

    This function reads audio data from an existing LMDB dataset, processes it using the HuBERT model,
    and stores the resulting audio embeddings in a new LMDB dataset.

    The function performs the following steps:
    1. Parses command-line arguments to get the path for saving the new LMDB dataset.
    2. Loads the pre-trained HuBERT model and processor.
    3. Opens the existing LMDB dataset and reads metadata such as the number of entries and shapes of various data components.
    4. Defines a helper function to decode bytes data into audio and other components.
    5. Iterates through the dataset in steps, processes the audio data to generate embeddings, and stores the embeddings in the new LMDB dataset.

    Args:
        None

    Returns:
        None
    """
    import soundfile as sf
    parser = argparse.ArgumentParser(description="Create LMDB dataset")
    parser.add_argument("lmdb_path", type=str, default="",
                        help="Path to save the lmdb")
    args = parser.parse_args()

    # model_id = "/mnt/4TDisk/mm_data/mm/huggingface/facebook/hubert-large-ls960-ft"
    # hubert_model = HubertModel.from_pretrained(model_id).requires_grad_(False).eval().cuda()
    model_id = MODEL_NAMES["wavlm"]
    hubert_model = WavLMModel.from_pretrained(model_id).requires_grad_(False).eval().cuda()
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_id)


    db = lmdb.open(args.lmdb_path[:-5] + "_wavlm.lmdb", map_size=2**39)
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

        Step = 2048
        N = num // Step
        for n in range(N):
            with db.begin(write=True) as txn_hubert:
                for k in tqdm(range(Step), total=Step):
                    dp_id = n * Step + k
                    bytes_data_item = txn.get(f"{dp_id:010d}".encode("utf-8"))
                    audio = DecodeFromBytes(bytes_data_item, seq_size=seq_size, betas_shape=betas_shape,
                                            poses_shape=poses_shape, expressions_shape=expressions_shape)[-1]
                    aud = wav2vec2_processor(
                        audio, return_tensors='pt', sampling_rate=16000).input_values
                    aud = aud.cuda()
                    with torch.no_grad():
                        audio_embeding = hubert_model.forward(
                            aud).last_hidden_state

                    audio_embeding = F.interpolate(audio_embeding.permute((0, 2, 1)), seq_size, mode='linear',
                                                   align_corners=True).permute((0, 2, 1))
                    audio_embeding = audio_embeding[0].reshape(
                        -1).cpu().numpy().tobytes()
                    txn_hubert.put(f"{dp_id:010d}".encode(
                        "utf-8"), audio_embeding)

        with db.begin(write=True) as txn_hubert:
            for k in range(N*Step, num):
                dp_id = k
                bytes_data = txn.get(f"{dp_id:010d}".encode("utf-8"))
                audio = DecodeFromBytes(bytes_data, seq_size=seq_size, betas_shape=betas_shape,
                                            poses_shape=poses_shape, expressions_shape=expressions_shape)[-1]
                aud = wav2vec2_processor(
                    audio, return_tensors='pt', sampling_rate=16000).input_values
                aud = aud.cuda()
                with torch.no_grad():
                    audio_embeding = hubert_model.forward(
                        aud).last_hidden_state

                audio_embeding = F.interpolate(audio_embeding.permute((0, 2, 1)), seq_size, mode='linear',
                                               align_corners=True).permute((0, 2, 1))
                audio_embeding = audio_embeding[0].reshape(
                    -1).cpu().numpy().tobytes()
                txn_hubert.put(f"{dp_id:010d}".encode("utf-8"), audio_embeding)


if __name__ == "__main__":
    create()
