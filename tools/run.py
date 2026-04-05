#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd, cwd):
    printable = " ".join(str(x) for x in cmd)
    print(f"[RUN] ({cwd}) {printable}")
    subprocess.run(cmd, cwd=cwd, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Unified entrypoint for GlobalDiff.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_lmdb = subparsers.add_parser("preprocess-lmdb", help="Create LMDB files from BEAT2.")
    preprocess_lmdb.add_argument("data_root", help="Path to beat_english_v2.0.0")
    preprocess_lmdb.add_argument("--split", default="train", choices=["all", "train", "test", "val", "additional"])
    preprocess_lmdb.add_argument("--seq-size", type=int, default=60)
    preprocess_lmdb.add_argument("--stride-size", type=int, default=20)
    preprocess_lmdb.add_argument("--local-rotation", action="store_true")
    preprocess_lmdb.add_argument("--lmdb-path", default="")

    preprocess_wavlm = subparsers.add_parser("preprocess-wavlm", help="Extract WavLM features into LMDB.")
    preprocess_wavlm.add_argument("lmdb_path", help="Path to a *_global.lmdb file")

    train_vae = subparsers.add_parser("train-vae", help="Launch VAE training.")
    train_vae.add_argument("--nproc-per-node", type=int, default=1)
    train_vae.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to Train_MaskedVAE.py")

    train_fm = subparsers.add_parser("train-fm", help="Launch GlobalDiff training.")
    train_fm.add_argument("--nproc-per-node", type=int, default=4)
    train_fm.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to TrainFixedExpressions.py")

    infer = subparsers.add_parser("infer", help="Run inference and internal FID evaluation.")
    infer.add_argument("ckpt_folder", help="Checkpoint folder under Scripts/FM/ckpt")
    infer.add_argument("--data-root", required=True, help="Path to beat_english_v2.0.0")
    infer.add_argument("--pid", default="2")
    infer.add_argument("--outdir", default="tmp_test_out")
    infer.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to Test_FixedExpressions_pid_batch.py")

    env = subparsers.add_parser("check-env", help="Run the environment checker.")
    env.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to tools/check_env.py")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess-lmdb":
        cmd = [
            sys.executable,
            "create_lmdb.py",
            args.data_root,
            "--split",
            args.split,
            "--seq_size",
            str(args.seq_size),
            "--stride_size",
            str(args.stride_size),
        ]
        if args.local_rotation:
            cmd.extend(["--local_rotation", "1"])
        if args.lmdb_path:
            cmd.extend(["--lmdb_path", args.lmdb_path])
        run_command(cmd, REPO_ROOT / "Data" / "BEAT2")
        return 0

    if args.command == "preprocess-wavlm":
        cmd = [sys.executable, "create_hubert.py", args.lmdb_path]
        run_command(cmd, REPO_ROOT / "Data" / "BEAT2")
        return 0

    if args.command == "train-vae":
        cmd = ["torchrun", "--nproc_per_node", str(args.nproc_per_node), "Train_MaskedVAE.py"] + args.extra
        run_command(cmd, REPO_ROOT / "Scripts" / "VAE")
        return 0

    if args.command == "train-fm":
        cmd = ["torchrun", "--nproc_per_node", str(args.nproc_per_node), "TrainFixedExpressions.py"] + args.extra
        run_command(cmd, REPO_ROOT / "Scripts" / "FM")
        return 0

    if args.command == "infer":
        cmd = [
            sys.executable,
            "Test_FixedExpressions_pid_batch.py",
            args.ckpt_folder,
            "--data_root",
            args.data_root,
            "--pid",
            args.pid,
            "--outdir",
            args.outdir,
        ] + args.extra
        run_command(cmd, REPO_ROOT / "Scripts" / "FM")
        return 0

    if args.command == "check-env":
        cmd = [sys.executable, str(REPO_ROOT / "tools" / "check_env.py")] + args.extra
        run_command(cmd, REPO_ROOT)
        return 0

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
