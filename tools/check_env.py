#!/usr/bin/env python3

import argparse
import importlib
import os
import sys
from pathlib import Path


REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "librosa",
    "soundfile",
    "lmdb",
    "tqdm",
    "einops",
    "tensorboard",
    "smplx",
    "transformers",
    "diffusers",
    "positional_encodings",
]


def _status(ok: bool, label: str, detail: str = ""):
    prefix = "[OK]" if ok else "[MISSING]"
    if detail:
        print(f"{prefix} {label}: {detail}")
    else:
        print(f"{prefix} {label}")


def _check_python():
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 10
    _status(ok, "Python", f"{version.major}.{version.minor}.{version.micro}")
    return ok


def _check_packages():
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            _status(True, f"package `{package}`")
        except Exception:
            missing.append(package)
            _status(False, f"package `{package}`")
    return missing


def _check_torch():
    try:
        import torch
    except Exception:
        _status(False, "package `torch`")
        return False

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    detail = f"version={torch.__version__}, cuda_available={cuda_available}, devices={device_count}"
    _status(True, "package `torch`", detail)
    return True


def _check_file(path: Path, label: str):
    ok = path.exists()
    _status(ok, label, str(path))
    return ok


def _resolve_smplx_path(repo_root: Path):
    env_path = os.environ.get("GLOBALDIFF_SMPLX_MODEL_PATH")
    if env_path:
        return Path(env_path)
    return repo_root / "Data" / "SMPLX_NEUTRAL_2020.npz"


def main():
    parser = argparse.ArgumentParser(description="Check whether GlobalDiff is ready to run.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional BEAT2 root, e.g. /path/to/beat_v2.0.0/beat_english_v2.0.0",
    )
    parser.add_argument(
        "--check-train-artifacts",
        action="store_true",
        help="Also check training-time artifacts such as LMDBs and pretrained checkpoints.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    print(f"Checking GlobalDiff in: {repo_root}")

    ok = True
    ok &= _check_python()
    ok &= _check_torch()
    missing_packages = _check_packages()
    ok &= not missing_packages

    smplx_path = _resolve_smplx_path(repo_root)
    ok &= _check_file(smplx_path, "SMPL-X model")

    simple_speech_zip = repo_root / "Scripts" / "FM" / "ckpt" / "split" / "SimpleSpeechModel" / "best.zip"
    simple_speech_pt = repo_root / "Scripts" / "FM" / "ckpt" / "split" / "SimpleSpeechModel" / "best.pt"
    vae2_ckpt = repo_root / "Scripts" / "VAE" / "ckpt" / "split" / "global" / "MaskedVAE2-HorizonFlip" / "best.pt"
    vae3_ckpt = repo_root / "Scripts" / "VAE" / "ckpt" / "split" / "global" / "MaskedVAE3-HorizonFlip" / "best.pt"

    ok &= _check_file(vae2_ckpt, "MaskedVAE2 checkpoint")
    ok &= _check_file(vae3_ckpt, "MaskedVAE3 checkpoint")
    ok &= _check_file(simple_speech_zip, "SimpleSpeechModel zip")
    if simple_speech_pt.exists():
        _status(True, "SimpleSpeechModel checkpoint", str(simple_speech_pt))
    else:
        _status(False, "SimpleSpeechModel checkpoint", "missing best.pt (unzip best.zip first)")

    wavlm_path = os.environ.get("GLOBALDIFF_WAVLM_PATH")
    if wavlm_path:
        _check_file(Path(wavlm_path), "WavLM local path")
    else:
        _status(True, "WavLM source", "using Hugging Face model id `patrickvonplaten/wavlm-libri-clean-100h-large`")

    if args.data_root is not None:
        data_root = args.data_root
        ok &= _check_file(data_root, "BEAT2 root")
        ok &= _check_file(data_root / "smplxflame_30", "BEAT2 smplxflame_30")
        ok &= _check_file(data_root / "wave16k", "BEAT2 wave16k")
        ok &= _check_file(data_root / "sem", "BEAT2 sem")
        ok &= _check_file(data_root / "train_test_split.csv", "BEAT2 train_test_split.csv")

    if args.check_train_artifacts:
        train_lmdb = repo_root / "Data" / "BEAT2" / "train_seq_size_60_stride_size_20_global.lmdb"
        train_wavlm_lmdb = repo_root / "Data" / "BEAT2" / "train_seq_size_60_stride_size_20_global_wavlm.lmdb"
        _check_file(train_lmdb, "Train LMDB")
        _check_file(train_wavlm_lmdb, "Train WavLM LMDB")

    if ok:
        print("Environment check passed.")
        return 0

    print("Environment check failed. Please fix the missing items above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
