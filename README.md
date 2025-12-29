


<div align="center">
<h2><font> </font></center> <br> <center>Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints</h2>

[Xiangyue Zhang\*](https://xiangyue-zhang.github.io/), [Jianfang Li\*†](https://github.com/Xiangyue-Zhang/SemTalk), [Jianqiang Ren](https://github.com/JianqiangRen), [Jiaxu Zhang](https://kebii.github.io/)

<p align="center">
  <strong>✨AAAI 2026✨</strong>
</p>
---

GlobalDiff is developed by Alibaba Cloud and released under the **Apache License 2.0**.

<img src="assets/teaser.jpg" alt="GlobalDiff Overview" width="750"/>
</div>

## 🚧 Code Release Plan
- Training: ✅ 2025.12.29
- Testing: ✅ 2025.12.29

---

# 💖 Inference Data

If you would like to compare your paper’s results with GlobalDiff but find it too difficult to run the repository, you can simply download the test `.npz` file. Results for **person-2** are provided in `best_pid_2.zip`.

## Download Data

please refer to [EMAGE](https://github.com/PantoMatrix/PantoMatrix/tree/main) and download datasets from [BEAT2](https://huggingface.co/datasets/H-Liu1997/BEAT2) for datasets.
If you are in China, you can use hf-mirror for faster and more reliable downloads. The process may take some time, so please be patient.
```shell
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download H-Liu1997/BEAT2 --local-dir H-Liu1997/BEAT2
```

## 1) Create Training Data

Go to `Data/BEAT2`, then run:

```shell
cd Data/BEAT2
```

Create LMDB:
```shell
    python create_lmdb.py path_to_folder/beat_v2.0.0/beat_english_v2.0.0/
```
This may take a while. Then extract HuBERT features:
```shell
    python create_hubert.py train_seq_size_60_stride_size_20_global.lmdb
```
This takes about **1 hour**.

You will get two folders:
- `train_seq_size_60_stride_size_20_global.lmdb`
- `train_seq_size_60_stride_size_20_global_wavlm.lmdb`

---

## 2) Training

Unzip `best.zip` to:

-    `Scripts/FM/ckpt/split/SimpleSpeechModel/`

Then run training:
```shell
    cd Scripts/FM
    torchrun --nproc_per_node=4 TrainFixedExpressions.py
```
---

## 3) Inference
```shell
    python Test_FixedExpressions_pid_batch.py \
      ckpt/split/DiffusionDITNetPartsFixedExpressions2PostNorm_LL_split_HorizonFlip_MaskedVAE3_W02_BoneDirLoss \
      --data_root=path_to_folder/beat_v2.0.0/beat_english_v2.0.0/
```
This will run **all checkpoints sequentially**. The FID record is saved to:

-    `ckpt/split/DiffusionDITNetPartsFixedExpressions2PostNorm_LL_split_HorizonFlip_MaskedVAE3_W02_BoneDirLoss/record_2.txt`

The results with the best FID are saved to:

-    `ckpt/split/DiffusionDITNetPartsFixedExpressions2PostNorm_LL_split_HorizonFlip_MaskedVAE3_W02_BoneDirLoss/best_pid_2/`

> **Note:** This FID is computed using our internal evaluation implementation.  
> For fair comparison, please use the official evaluation from **EMAGE** or **SemTalk**:
> - EMAGE: https://pantomatrix.github.io/EMAGE/
> - SemTalk: https://github.com/Xiangyue-Zhang/SemTalk

---

## 4) 📺 Visualization

Following [EMAGE](https://github.com/PantoMatrix/PantoMatrix), you can download [SMPLX blender addon](https://huggingface.co/datasets/H-Liu1997/BEAT2_Tools/blob/main/smplx_blender_addon_20230921.zip), and install it in your blender 3.x or 4.x. Click the button Add Animation to visualize the generated smplx file (like xxx.npz).

# 🙏 Acknowledgments
Thanks to [EMAGE](https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024), [SemTalk](https://github.com/Xiangyue-Zhang/SemTalk), our code is partially borrowing from them. Please check these useful repos.

## 📚 Citation
If you find our code or paper helps, please consider citing:
```bibtex
    @article{zhang2025mitigating,
      title={Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints},
      author={Zhang, Xiangyue and Li, Jianfang and Ren, Jianqiang and Zhang, Jiaxu},
      journal={arXiv preprint arXiv:2511.10076},
      year={2025}
    }
```
