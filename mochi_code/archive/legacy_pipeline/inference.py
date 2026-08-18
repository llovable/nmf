#!/usr/bin/env python3
"""
멀티오믹스 NMF+TGAN 추론 스크립트
- 학습된 모델을 사용하여 결측치 보정
- 결과를 TSV 파일로 저장
"""

import os
import torch
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import json

# 로컬 모듈 임포트
from models import Generator
from dataloader import get_multi_omics_dataloaders
from utils import calculate_imputation_metrics

@torch.no_grad()
def inference(
    model: Generator,
    data_loader,
    device: torch.device,
    output_path: str
):
    """
    모델 추론 및 결과 저장
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_masks = []
    all_samples = []
    
    print("🔍 추론 중...")
    
    for batch in data_loader:
        x_src = batch['x_src'].to(device)
        y_true = batch['y_true'].to(device)
        mask = batch['mask'].to(device)
        samples = batch['samples']
        
        # 예측 (self-attention)
        y_pred = model(x_src, src=None)
        
        all_predictions.append(y_pred.cpu().numpy())
        all_targets.append(y_true.cpu().numpy())
        all_masks.append(mask.cpu().numpy())
        all_samples.extend(samples)
    
    # 결과 합치기
    y_pred_all = np.concatenate(all_predictions, axis=0)
    y_true_all = np.concatenate(all_targets, axis=0)
    mask_all = np.concatenate(all_masks, axis=0)
    
    # 성능 평가
    metrics = calculate_imputation_metrics(y_true_all, y_pred_all, mask_all)
    
    print(f"✅ 추론 완료!")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Pearson: {metrics['pearson']:.4f}")
    print(f"결측치 수: {metrics['n_missing']}")
    
    # 결과를 DataFrame으로 변환
    # 예측값으로 결측치 채우기
    y_filled = y_true_all.copy()
    y_filled[mask_all.astype(bool)] = y_pred_all[mask_all.astype(bool)]
    
    # 결과 저장
    result_df = pd.DataFrame(
        y_filled,
        index=all_samples,
        columns=[f"feature_{i}" for i in range(y_filled.shape[1])]
    )
    
    result_df.to_csv(output_path, sep='\t')
    print(f"💾 결과 저장: {output_path}")
    
    return metrics, result_df

def main():
    parser = argparse.ArgumentParser(description='멀티오믹스 NMF+TGAN 추론')
    
    # 모델 설정
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='학습된 모델 체크포인트 경로')
    
    # 데이터 설정
    parser.add_argument('--data_dir', type=str, default='./processed_datasets',
                       help='전처리된 데이터 디렉토리')
    parser.add_argument('--src_modality', type=str, default='rna',
                       help='소스 모달리티 (rna, methyl, protein)')
    parser.add_argument('--tgt_modality', type=str, default='protein',
                       help='타깃 모달리티 (rna, methyl, protein)')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--device', type=str, default='auto',
                       help='디바이스 (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 멀티오믹스 NMF+TGAN 추론 시작")
    print(f"체크포인트: {args.checkpoint}")
    print(f"디바이스: {device}")
    print(f"소스: {args.src_modality} → 타깃: {args.tgt_modality}")
    
    # 체크포인트 로드
    print("\n📂 체크포인트 로딩 중...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        print(f"✅ 체크포인트 로드 성공 (에포크 {checkpoint['epoch']})")
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        return
    
    # 데이터로더 생성
    print("\n📊 데이터 로딩 중...")
    try:
        train_loader, valid_loader, _, data_info = get_multi_omics_dataloaders(
            data_dir=args.data_dir,
            src_modality=args.src_modality,
            tgt_modality=args.tgt_modality,
            batch_size=32  # 추론용 배치 크기
        )
        print(f"✅ 데이터 로딩 성공!")
        print(f"소스 차원: {data_info['src_dim']}")
        print(f"타깃 차원: {data_info['tgt_dim']}")
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return
    
    # 모델 생성
    print("\n🤖 모델 생성 중...")
    
    target_type = args.tgt_modality
    
    generator = Generator(
        input_size=data_info['src_dim'],
        output_size=data_info['tgt_dim'],
        use_attn=True,  # 체크포인트에서 확인
        n_heads=4,
        d_head=64,
        n_src_tokens=8,
        target_type=target_type,
        src_size=data_info['src_dim']
    )
    
    # 체크포인트에서 모델 가중치 로드
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    
    print(f"✅ 모델 생성 완료!")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 추론 실행
    print("\n🔍 추론 실행 중...")
    
    # 훈련 데이터 추론
    train_output_path = os.path.join(args.output_dir, f'{args.tgt_modality}_train_imputed.tsv')
    train_metrics, train_df = inference(generator, train_loader, device, train_output_path)
    
    # 검증 데이터 추론
    valid_output_path = os.path.join(args.output_dir, f'{args.tgt_modality}_valid_imputed.tsv')
    valid_metrics, valid_df = inference(generator, valid_loader, device, valid_output_path)
    
    # 전체 결과 요약
    summary = {
        'train': train_metrics,
        'valid': valid_metrics,
        'model_info': {
            'checkpoint': args.checkpoint,
            'src_modality': args.src_modality,
            'tgt_modality': args.tgt_modality,
            'src_dim': data_info['src_dim'],
            'tgt_dim': data_info['tgt_dim']
        }
    }
    
    # 요약 저장
    summary_path = os.path.join(args.output_dir, 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 추론 완료!")
    print(f"결과 저장 위치: {args.output_dir}")
    print(f"훈련 데이터 RMSE: {train_metrics['rmse']:.4f}")
    print(f"검증 데이터 RMSE: {valid_metrics['rmse']:.4f}")

if __name__ == '__main__':
    main()
