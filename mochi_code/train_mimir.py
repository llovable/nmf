#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MIMIR phase1+2를 BRCA 631 스플릿에서 학습한다."""

import argparse
from pathlib import Path

import torch

from mimir_wrap import train_mimir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--save_dir", default="/home/dyan/nmf/mochi_code/results/current/mimir")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--phase1_epochs", type=int, default=70)
    ap.add_argument("--phase2_epochs", type=int, default=120)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    train_mimir(args.data_dir, args.save_dir, device,
                phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs)


if __name__ == "__main__":
    main()
