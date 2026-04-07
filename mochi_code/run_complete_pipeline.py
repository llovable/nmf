#!/usr/bin/env python3
"""
멀티오믹스 NMF+TGAN 전체 파이프라인
1. 데이터 전처리 (데이터 생성 및 전처리)
2. 모델 학습
3. 데이터셋 결측치 imputation 수행
4. 다운스트림 분석을 통한 성능평가
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# 로컬 모듈 임포트
from models import Generator
from dataloader import get_multi_omics_dataloaders
from utils import evaluate_model_performance, calculate_imputation_metrics

class MultiOmicsPipeline:
    """
    멀티오믹스 NMF+TGAN 전체 파이프라인 클래스
    """
    
    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"{config['output_dir']}/pipeline_results_{self.timestamp}"
        
        # 결과 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 파이프라인 로그
        self.pipeline_log = []
        self.log_step("파이프라인 시작", config)
        
    def log_step(self, step, details=None):
        """파이프라인 단계 로깅"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'details': details
        }
        self.pipeline_log.append(log_entry)
        print(f"\n{'='*60}")
        print(f"🔄 {step}")
        print(f"{'='*60}")
        if details:
            print(f"설정: {details}")
    
    def run_command(self, command, description):
        """명령어 실행 및 결과 출력"""
        print(f"\n🚀 {description}")
        print(f"명령어: {command}")
        print("-" * 50)
        
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print("✅ 성공!")
            if result.stdout:
                print("출력:")
                print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 실패: {e}")
            if e.stdout:
                print("표준 출력:")
                print(e.stdout)
            if e.stderr:
                print("오류 출력:")
                print(e.stderr)
            return False
    
    def stage1_data_preprocessing(self):
        """1단계: 데이터 전처리 (데이터 생성 및 전처리)"""
        self.log_step("1단계: 데이터 전처리 시작")
        
        # 기존 전처리 스크립트 실행
        if self.config.get('skip_preprocessing', False):
            print("⏭️ 데이터 전처리 건너뛰기")
            return True
        
        # 향상된 데이터 전처리 스크립트 실행
        prep_script = "data_prep_enhanced.py"
        if os.path.exists(prep_script):
            prep_command = f"python {prep_script} \
                --rna {self.config['raw_data_dir']}/TCGA-BRCA.star_tpm.tsv \
                --methy {self.config['raw_data_dir']}/TCGA-BRCA.methylation450.tsv \
                --protein {self.config['raw_data_dir']}/TCGA-BRCA.protein.tsv \
                --subtype {self.config['subtype_file']} \
                --out_dir {self.config['processed_data_dir']} \
                --prefix BRCA_PAM50"
            
            if not self.run_command(prep_command, "향상된 데이터 전처리 실행"):
                return False
        else:
            print(f"❌ 전처리 스크립트를 찾을 수 없습니다: {prep_script}")
            return False
        
        # 전처리 결과 확인
        processed_dir = self.config['processed_data_dir']
        required_files = [
            f"{self.config['src_modality']}.tsv",
            f"{self.config['tgt_modality']}.tsv", 
            f"{self.config['tgt_modality']}_mask.tsv"
        ]
        
        for file in required_files:
            file_path = os.path.join(processed_dir, file)
            if not os.path.exists(file_path):
                print(f"❌ 필요한 파일이 없습니다: {file_path}")
                return False
        
        print("✅ 데이터 전처리 완료!")
        return True
    
    def stage2_model_training(self):
        """2단계: 모델 학습"""
        self.log_step("2단계: 모델 학습 시작")
        
        if self.config.get('skip_training', False):
            print("⏭️ 모델 학습 건너뛰기")
            return True
        
        # 학습 명령어 구성
        train_command = f"python train_multi_omics.py \
            --data_dir {self.config['processed_data_dir']} \
            --src_modality {self.config['src_modality']} \
            --tgt_modality {self.config['tgt_modality']} \
            --use_attention \
            --n_heads {self.config.get('n_heads', 4)} \
            --d_head {self.config.get('d_head', 64)} \
            --n_src_tokens {self.config.get('n_src_tokens', 8)} \
            --epochs {self.config['epochs']} \
            --batch_size {self.config['batch_size']} \
            --lr_g {self.config.get('lr_g', 0.0001)} \
            --lr_d {self.config.get('lr_d', 0.0001)} \
            --lambda_gp {self.config.get('lambda_gp', 10.0)} \
            --n_critic {self.config.get('n_critic', 5)} \
            --output_dir {self.results_dir}/training"
        
        if not self.run_command(train_command, "멀티오믹스 모델 학습"):
            return False
        
        # 학습 결과 확인
        best_model_path = f"{self.results_dir}/training/model_best.ckpt"
        if not os.path.exists(best_model_path):
            print(f"❌ 최고 성능 모델을 찾을 수 없습니다: {best_model_path}")
            return False
        
        print("✅ 모델 학습 완료!")
        return True
    
    def stage3_imputation(self):
        """3단계: 데이터셋 결측치 imputation 수행"""
        self.log_step("3단계: 결측치 imputation 시작")
        
        if self.config.get('skip_imputation', False):
            print("⏭️ Imputation 건너뛰기")
            return True
        
        # Imputation 명령어 구성
        impute_command = f"python inference.py \
            --checkpoint {self.results_dir}/training/model_best.ckpt \
            --data_dir {self.config['processed_data_dir']} \
            --src_modality {self.config['src_modality']} \
            --tgt_modality {self.config['tgt_modality']} \
            --output_dir {self.results_dir}/imputation"
        
        if not self.run_command(impute_command, "결측치 imputation 실행"):
            return False
        
        # Imputation 결과 확인
        impute_dir = f"{self.results_dir}/imputation"
        required_files = [
            f"{self.config['tgt_modality']}_train_imputed.tsv",
            f"{self.config['tgt_modality']}_valid_imputed.tsv",
            "inference_summary.json"
        ]
        
        for file in required_files:
            file_path = os.path.join(impute_dir, file)
            if not os.path.exists(file_path):
                print(f"❌ 필요한 파일이 없습니다: {file_path}")
                return False
        
        print("✅ 결측치 imputation 완료!")
        return True
    
    def stage4_downstream_analysis(self):
        """4단계: 다운스트림 분석을 통한 성능평가"""
        self.log_step("4단계: 다운스트림 분석 시작")
        
        if self.config.get('skip_downstream', False):
            print("⏭️ 다운스트림 분석 건너뛰기")
            return True
        
        try:
            # Imputation 결과 로드
            impute_dir = f"{self.results_dir}/imputation"
            train_imputed = pd.read_csv(
                os.path.join(impute_dir, f"{self.config['tgt_modality']}_train_imputed.tsv"), 
                sep='\t', index_col=0
            )
            valid_imputed = pd.read_csv(
                os.path.join(impute_dir, f"{self.config['tgt_modality']}_valid_imputed.tsv"), 
                sep='\t', index_col=0
            )
            
            # 원본 데이터 로드 (비교용)
            original_data = pd.read_csv(
                os.path.join(self.config['processed_data_dir'], f"{self.config['tgt_modality']}.tsv"),
                sep='\t', index_col=0
            )
            
            # 마스크 로드
            mask_data = pd.read_csv(
                os.path.join(self.config['processed_data_dir'], f"{self.config['tgt_modality']}_mask.tsv"),
                sep='\t', index_col=0
            )
            
            print(f"✅ 데이터 로드 완료")
            print(f"훈련 데이터: {train_imputed.shape}")
            print(f"검증 데이터: {valid_imputed.shape}")
            print(f"원본 데이터: {original_data.shape}")
            
            # 성능 분석
            analysis_results = self.perform_downstream_analysis(
                train_imputed, valid_imputed, original_data, mask_data
            )
            
            # 결과 저장
            analysis_path = f"{self.results_dir}/downstream_analysis"
            os.makedirs(analysis_path, exist_ok=True)
            
            # 분석 결과 저장
            with open(f"{analysis_path}/analysis_results.json", 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            # 성능 요약 저장
            summary_df = pd.DataFrame([analysis_results])
            summary_df.to_csv(f"{analysis_path}/performance_summary.tsv", sep='\t', index=False)
            
            print("✅ 다운스트림 분석 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 다운스트림 분석 실패: {e}")
            return False
    
    def perform_downstream_analysis(self, train_imputed, valid_imputed, original_data, mask_data):
        """다운스트림 성능 분석 수행"""
        
        # 공통 샘플 찾기
        common_samples = train_imputed.index.intersection(valid_imputed.index).intersection(original_data.index)
        
        # 데이터 정렬
        train_aligned = train_imputed.loc[common_samples]
        valid_aligned = valid_imputed.loc[common_samples]
        original_aligned = original_data.loc[common_samples]
        mask_aligned = mask_data.loc[common_samples]
        
        # 성능 지표 계산
        results = {
            'n_samples': len(common_samples),
            'n_features': train_aligned.shape[1],
            'src_modality': self.config['src_modality'],
            'tgt_modality': self.config['tgt_modality'],
            'timestamp': self.timestamp
        }
        
        # 1. Imputation 성능 (결측치 위치에서만)
        mask_bool = mask_aligned.astype(bool)
        
        # 훈련 데이터 성능
        train_metrics = calculate_imputation_metrics(
            original_aligned.values, train_aligned.values, mask_aligned.values
        )
        results['train_imputation'] = train_metrics
        
        # 검증 데이터 성능
        valid_metrics = calculate_imputation_metrics(
            original_aligned.values, valid_aligned.values, mask_aligned.values
        )
        results['valid_imputation'] = valid_metrics
        
        # 2. 전체 데이터 품질 (결측치 포함)
        # 결측치 비율
        missing_ratio = mask_bool.sum().sum() / mask_bool.size
        results['missing_ratio'] = float(missing_ratio)
        
        # 3. 데이터 분포 분석
        # 원본 vs Imputed 비교
        train_corr = np.corrcoef(original_aligned.values.flatten(), train_aligned.values.flatten())[0, 1]
        valid_corr = np.corrcoef(original_aligned.values.flatten(), valid_aligned.values.flatten())[0, 1]
        
        results['overall_correlation'] = {
            'train': float(train_corr) if not np.isnan(train_corr) else None,
            'valid': float(valid_corr) if not np.isnan(valid_corr) else None
        }
        
        # 4. 특징별 성능 분석
        feature_performance = []
        for i in range(original_aligned.shape[1]):
            col_mask = mask_aligned.iloc[:, i].astype(bool)
            if col_mask.sum() > 0:  # 결측치가 있는 특징만
                col_metrics = calculate_imputation_metrics(
                    original_aligned.iloc[:, i].values.reshape(-1, 1),
                    train_aligned.iloc[:, i].values.reshape(-1, 1),
                    mask_aligned.iloc[:, i].values.reshape(-1, 1)
                )
                feature_performance.append({
                    'feature_id': i,
                    'n_missing': int(col_mask.sum()),
                    'rmse': col_metrics['rmse'],
                    'mae': col_metrics['mae'],
                    'pearson': col_metrics['pearson']
                })
        
        results['feature_performance'] = feature_performance
        
        return results
    
    def run_pipeline(self):
        """전체 파이프라인 실행"""
        print(f"🚀 멀티오믹스 NMF+TGAN 전체 파이프라인 시작")
        print(f"결과 저장 위치: {self.results_dir}")
        print(f"설정: {json.dumps(self.config, indent=2)}")
        
        start_time = time.time()
        
        # 1단계: 데이터 전처리
        if not self.stage1_data_preprocessing():
            print("❌ 1단계 실패. 파이프라인을 중단합니다.")
            return False
        
        # 2단계: 모델 학습
        if not self.stage2_model_training():
            print("❌ 2단계 실패. 파이프라인을 중단합니다.")
            return False
        
        # 3단계: Imputation
        if not self.stage3_imputation():
            print("❌ 3단계 실패. 파이프라인을 중단합니다.")
            return False
        
        # 4단계: 다운스트림 분석
        if not self.stage4_downstream_analysis():
            print("❌ 4단계 실패. 파이프라인을 중단합니다.")
            return False
        
        # 파이프라인 완료
        end_time = time.time()
        duration = end_time - start_time
        
        self.log_step("파이프라인 완료", {
            'duration_seconds': duration,
            'duration_minutes': duration / 60,
            'results_dir': self.results_dir
        })
        
        # 최종 로그 저장
        with open(f"{self.results_dir}/pipeline_log.json", 'w') as f:
            json.dump(self.pipeline_log, f, indent=2)
        
        print(f"\n🎉 파이프라인 완료!")
        print(f"소요 시간: {duration/60:.1f}분")
        print(f"결과 저장 위치: {self.results_dir}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='멀티오믹스 NMF+TGAN 전체 파이프라인')
    
    # 데이터 경로 설정
    parser.add_argument('--raw_data_dir', type=str, default='/home/dyan/data/data',
                       help='원본 데이터 디렉토리')
    parser.add_argument('--subtype_file', type=str, default='/home/dyan/data/data/brca-2Fpam50.tsv',
                       help='PAM50 서브타입 파일 경로')
    parser.add_argument('--processed_data_dir', type=str, default='./processed_datasets',
                       help='전처리된 데이터 저장 디렉토리')
    parser.add_argument('--output_dir', type=str, default='./pipeline_results',
                       help='결과 저장 디렉토리')
    
    # 모달리티 설정
    parser.add_argument('--src_modality', type=str, default='rna',
                       help='소스 모달리티 (rna, methyl, protein)')
    parser.add_argument('--tgt_modality', type=str, default='protein',
                       help='타깃 모달리티 (rna, methyl, protein)')
    
    # 모델 설정
    parser.add_argument('--n_heads', type=int, default=4,
                       help='어텐션 헤드 수')
    parser.add_argument('--d_head', type=int, default=64,
                       help='각 헤드의 차원')
    parser.add_argument('--n_src_tokens', type=int, default=8,
                       help='소스 가상 토큰 수')
    
    # 학습 설정
    parser.add_argument('--epochs', type=int, default=100,
                       help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--lr_g', type=float, default=0.0001,
                       help='Generator 학습률')
    parser.add_argument('--lr_d', type=float, default=0.0001,
                       help='Critic 학습률')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                       help='Gradient Penalty 가중치')
    parser.add_argument('--n_critic', type=int, default=5,
                       help='Critic 업데이트 횟수')
    
    # 실행 옵션
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='데이터 전처리 건너뛰기')
    parser.add_argument('--skip_training', action='store_true',
                       help='모델 학습 건너뛰기')
    parser.add_argument('--skip_imputation', action='store_true',
                       help='Imputation 건너뛰기')
    parser.add_argument('--skip_downstream', action='store_true',
                       help='다운스트림 분석 건너뛰기')
    
    args = parser.parse_args()
    
    # 설정 딕셔너리 구성
    config = {
        'raw_data_dir': args.raw_data_dir,
        'subtype_file': args.subtype_file,
        'processed_data_dir': args.processed_data_dir,
        'output_dir': args.output_dir,
        'src_modality': args.src_modality,
        'tgt_modality': args.tgt_modality,
        'n_heads': args.n_heads,
        'd_head': args.d_head,
        'n_src_tokens': args.n_src_tokens,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr_g': args.lr_g,
        'lr_d': args.lr_d,
        'lambda_gp': args.lambda_gp,
        'n_critic': args.n_critic,
        'skip_preprocessing': args.skip_preprocessing,
        'skip_training': args.skip_training,
        'skip_imputation': args.skip_imputation,
        'skip_downstream': args.skip_downstream
    }
    
    # 파이프라인 실행
    pipeline = MultiOmicsPipeline(config)
    success = pipeline.run_pipeline()
    
    if success:
        print("\n🎉 전체 파이프라인이 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n❌ 파이프라인 실행 중 오류가 발생했습니다.")
        sys.exit(1)

if __name__ == '__main__':
    main()
