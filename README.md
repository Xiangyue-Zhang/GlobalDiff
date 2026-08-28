


<div align="center">
<h2><font> </font></center> <br> <center>Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints</h2>

[Xiangyue Zhang\*](https://xiangyue-zhang.github.io/), [Jianfang Li\*†](https://github.com/Xiangyue-Zhang/SemTalk), [Jianqiang Ren](https://github.com/JianqiangRen), [Jiaxu Zhang](https://kebii.github.io/)

<p align="center">
  <strong>✨AAAI 2026✨</strong>
</p>

<a href='https://arxiv.org/abs/2511.10076'><img src='https://img.shields.io/badge/ArXiv-2511.10076-red'></a> <a href='https://huggingface.co/papers/2511.10076'><img src='https://img.shields.io/badge/Hugging_Face-Paper-yellow'></a> <a href='https://xiangyuezhang.com/GlobalDiff/'><img src='https://img.shields.io/badge/Project-Page-purple'></a> <a href='https://huggingface.co/X-Zhang/GlobalDiff'><img src='https://img.shields.io/badge/%F0%9F%A4%97-Model_Weights-yellow'></a> <a href='https://huggingface.co/datasets/X-Zhang/GlobalDiff-Inference-Data/resolve/main/best_pid_2.zip'><img src='https://img.shields.io/badge/%F0%9F%A4%97-Speaker_2_Data-yellow'></a> <a href='https://drive.google.com/file/d/1FT5JyPKiHSimpy4imusYLs_Haq4JSd3o/view?usp=drive_link'><img src='https://img.shields.io/badge/Google_Drive-All_Speakers_Data-blue'></a>

<p><strong>Official implementation and checkpoints for long-horizon co-speech gesture generation with global rotation diffusion.</strong></p>

<p><sub>Keywords: co-speech gesture generation, co-speech motion generation, speech-driven motion generation, diffusion models, global rotation, long-horizon generation, BEAT2, and SMPL-X.</sub></p>

GlobalDiff is developed by Alibaba Cloud and released under the **Apache License 2.0**.

<img src="assets/teaser.jpg" alt="GlobalDiff Overview" width="750"/>
</div>

## 🚧 Code Release Plan
- Training: ✅ 2025.12.29
- Testing: ✅ 2025.12.29

---

# 💖 Inference Data

To compare against GlobalDiff without rerunning inference, download the generated test `.npz` files for the BEAT2 protocol used by your experiment:

| Evaluation setting | Coverage | Download |
| --- | --- | --- |
| **Speaker 2 (paper protocol)** | BEAT2 person ID 2 | [Hugging Face (`best_pid_2.zip`)](https://huggingface.co/datasets/X-Zhang/GlobalDiff-Inference-Data/resolve/main/best_pid_2.zip) |
| **All Speakers** | BEAT2 25-English-speaker setting | [Google Drive](https://drive.google.com/file/d/1FT5JyPKiHSimpy4imusYLs_Haq4JSd3o/view?usp=drive_link) |

Please select the archive that matches your evaluation setting. The Speaker 2 and All-Speakers results use different training and evaluation protocols, so they should not be interpreted as a controlled single-speaker versus multi-speaker ablation.

---

# 🤗 Model Weights

The official checkpoints are mirrored in the
[GlobalDiff Hugging Face repository](https://huggingface.co/X-Zhang/GlobalDiff).
It preserves the repository-relative paths for the global-rotation VAEs,
speech model, and evaluation representation asset. See the model card for the
complete file list and SHA-256 verification instructions.

---

# ⚡ Quick Start

GlobalDiff now includes two helper entrypoints:

- `python tools/check_env.py`: verify whether your local environment is ready
- `python tools/run.py ...`: unified commands for preprocessing, training, and inference

The fastest path to a runnable setup is:

```shell
conda create -n globaldiff python=3.10 -y
conda activate globaldiff
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python tools/check_env.py
```

If you already have BEAT2 and the required checkpoints prepared, you can run inference with:

```shell
python tools/run.py infer \
  ckpt/split/DiffusionDITNetPartsFixedExpressions2PostNorm_LL_split_HorizonFlip_MaskedVAE3_W02_BoneDirLoss \
  --data-root /path/to/beat_v2.0.0/beat_english_v2.0.0
```

---

# 🛠️ Environment Setup

The repository now provides a lightweight `requirements.txt`, a unified CLI under `tools/run.py`, and an environment checker under `tools/check_env.py` to make the project easier to reproduce.

## Recommended Setup

- OS: Linux
- Python: 3.10
- CUDA: 11.8 or a version compatible with your local PyTorch build
- GPU: at least 1 NVIDIA GPU for inference, 4 GPUs are recommended for the training command used below

Create a conda environment:

```shell
conda create -n globaldiff python=3.10 -y
conda activate globaldiff
```

Install PyTorch first. Please choose the command that matches your CUDA version from the official PyTorch website. For CUDA 11.8, you can use:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining Python dependencies:

```shell
pip install -r requirements.txt
```

## External Resources

GlobalDiff also relies on two external resources:

1. **SMPL-X model file**  
   Download `SMPLX_NEUTRAL_2020.npz` from the official SMPL-X release, then either:
   - place it at `Data/SMPLX_NEUTRAL_2020.npz`, or
   - set `GLOBALDIFF_SMPLX_MODEL_PATH` to its absolute path

2. **WavLM checkpoint**  
   By default, the code now uses the Hugging Face model id `patrickvonplaten/wavlm-libri-clean-100h-large`.  
   If you already have the model cached locally, you can point the code to that directory with:

```shell
export GLOBALDIFF_SMPLX_MODEL_PATH=/path/to/SMPLX_NEUTRAL_2020.npz
export GLOBALDIFF_WAVLM_PATH=/path/to/wavlm-libri-clean-100h-large
```

If you do not set `GLOBALDIFF_WAVLM_PATH`, Transformers will download the model automatically from Hugging Face the first time it is used.

## Quick Check

You can run:

```shell
python tools/check_env.py
```

or, if you want to verify the BEAT2 path as well:

```shell
python tools/check_env.py --data-root /path/to/beat_v2.0.0/beat_english_v2.0.0
```

The checker validates Python, core packages, PyTorch/CUDA, SMPL-X, WavLM source, and required checkpoints.

Before training or inference, make sure the following are ready:

- `Data/BEAT2/create_lmdb.py` can access your downloaded BEAT2 data
- `Scripts/VAE/ckpt/split/global/MaskedVAE2-HorizonFlip/best.pt` exists
- `Scripts/VAE/ckpt/split/global/MaskedVAE3-HorizonFlip/best.pt` exists
- `Scripts/FM/ckpt/split/SimpleSpeechModel/best.pt` is available after unzipping `best.zip`

## Download Data

please refer to [EMAGE](https://github.com/PantoMatrix/PantoMatrix/tree/main) and download datasets from [BEAT2](https://huggingface.co/datasets/H-Liu1997/BEAT2) for datasets.
If you are in China, you can use hf-mirror for faster and more reliable downloads. The process may take some time, so please be patient.
```shell
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download H-Liu1997/BEAT2 --local-dir H-Liu1997/BEAT2
```

## 1) Create Training Data

Create LMDB with the helper CLI:

```shell
python tools/run.py preprocess-lmdb /path/to/beat_v2.0.0/beat_english_v2.0.0 --split train
```

This may take a while. Then extract HuBERT features:

```shell
python tools/run.py preprocess-wavlm Data/BEAT2/train_seq_size_60_stride_size_20_global.lmdb
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
python tools/run.py train-fm --nproc-per-node 4
```

If you need custom arguments, append them after `--`, for example:

```shell
python tools/run.py train-fm --nproc-per-node 4 -- --batch_size 64 --epoch 500
```

---

## 3) Inference (Speaker 2 Paper Protocol)

The command below reproduces the Speaker 2 evaluation workflow. If you only need generated outputs, or if you are evaluating the 25-English-speaker setting, use the corresponding archive in [Inference Data](#-inference-data).

```shell
python tools/run.py infer \
  ckpt/split/DiffusionDITNetPartsFixedExpressions2PostNorm_LL_split_HorizonFlip_MaskedVAE3_W02_BoneDirLoss \
  --data-root /path/to/beat_v2.0.0/beat_english_v2.0.0
```
This will run **all checkpoints sequentially**. The FID record is saved to:

-    `ckpt/split/DiffusionDITNetPartsFixedExpressions2PostNorm_LL_split_HorizonFlip_MaskedVAE3_W02_BoneDirLoss/record_2.txt`

The results with the best FID are saved to:

-    `ckpt/split/DiffusionDITNetPartsFixedExpressions2PostNorm_LL_split_HorizonFlip_MaskedVAE3_W02_BoneDirLoss/best_pid_2/`

> **Note:** This FID is computed using our internal evaluation implementation.  
> For fair comparison, please use the official evaluation from **EMAGE** or **SemTalk**:
> - EMAGE: https://pantomatrix.github.io/EMAGE/
> - SemTalk: https://github.com/Xiangyue-Zhang/SemTalk

## 4) Legacy Scripts

The original research scripts are still preserved under `Data/BEAT2`, `Scripts/VAE`, and `Scripts/FM`.
If you prefer the old workflow, you can still call them directly. The new `tools/run.py` wrapper is only a convenience layer for easier reproduction.

---

## 5) 📺 Visualization

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
