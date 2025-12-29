AAAI2026 paper:Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints

GlobalDiff is developed by Alibaba Cloud and licensed under the Apache License (Version 2.0)




# create train data

 cd Data/BEAT2 then run following command:
```python
python create_lmdb.py path_to_folder/beat_v2.0.0/beat_english_v2.0.0/
 ```
 it takes a while to finish. then run 
 ```python
python create_hubert.py train_seq_size_60_stride_size_20_global.lmdb
```
it takes about one hour to finish.

you will get two folders: train_seq_size_60_stride_size_20_global.lmdb and train_seq_size_60_stride_size_20_global_wavlm.lmdb

# train
unzip best.zip in Scripts/FM/ckpt/split/SimpleSpeechModel/
```bash
cd Scripts/FM


torchrun --nproc_per_node=4 TrainFixedExpressions.py
```

# inference 
```python
python Test_FixedExpressions_pid_batch.py   ckpt/split/DiffusionDITNetPartsFixedExpressions2PostNorm_LL_split_HorizonFlip_MaskedVAE3_W02_BoneDirLoss  --data_root=path_to_folder/beat_v2.0.0/beat_english_v2.0.0/
```

this will inference the all the checkpoints sequentially. the fid is stored in 
```bash
ckpt/split/DiffusionDITNetPartsFixedExpressions2PostNorm_LL_split_HorizonFlip_MaskedVAE3_W02_BoneDirLoss/record_2.txt
```

the results with best fid is stored in  
```bash
ckpt/split/DiffusionDITNetPartsFixedExpressions2PostNorm_LL_split_HorizonFlip_MaskedVAE3_W02_BoneDirLoss/best_pid_2/
```
Note that this fid is based on our implementations to assess each checkpoint.
Please follow the original [EMAGE](https://pantomatrix.github.io/EMAGE/)  or [SemTalk](https://github.com/Xiangyue-Zhang/SemTalk)to calculate the fid for a fair comparison. 


# Visualize
follow [EMAGE](https://pantomatrix.github.io/EMAGE/) or [SemTalk](https://github.com/Xiangyue-Zhang/SemTalk) to use the SMPLX blender addon to visualize the npz results.

# results 
The results for person-2 are in best_pid_2.zip
 
