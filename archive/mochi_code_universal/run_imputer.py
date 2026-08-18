#!/usr/bin/env python3
"""
사용자 선택 모델로 결측치 보간 실행.
입력: TSV (특징 x 샘플)
"""
from __future__ import annotations

import argparse
import os
import json
from typing import Dict

import pandas as pd

from imputers.wrapper import impute, list_models


def _read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def main():
    parser = argparse.ArgumentParser(description="결측 보간 실행 스크립트")
    parser.add_argument("--input", required=True, help="입력 TSV (features x samples)")
    parser.add_argument("--output", required=True, help="출력 TSV 경로")
    parser.add_argument("--model", required=True, help="모델 이름 (mean|median|knn|softimpute|missforest)")
    parser.add_argument("--params", default=None, help="모델 파라미터 JSON 문자열")
    parser.add_argument("--params_file", default=None, help="모델 파라미터 JSON 파일 경로")
    parser.add_argument("--list_models", action="store_true", help="지원 모델 목록 출력")

    args = parser.parse_args()

    if args.list_models:
        models = list_models()
        print("지원 모델:")
        for k, v in models.items():
            print(f"  - {k}: {v}")
        return

    params: Dict = {}
    if args.params:
        params.update(json.loads(args.params))
    if args.params_file:
        with open(args.params_file, "r") as f:
            params.update(json.load(f))

    df = _read_tsv(args.input)
    out = impute(df, model=args.model, **params)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, sep="\t")
    print(f"✅ 완료: {args.output}")


if __name__ == "__main__":
    main()
