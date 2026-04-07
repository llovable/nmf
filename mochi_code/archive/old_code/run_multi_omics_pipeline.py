#!/usr/bin/env python3
"""
멀티오믹스 NMF+TGAN 전체 파이프라인 실행 스크립트
1. 데이터 전처리
2. 3가지 데이터셋 생성 (원본, 노이즈, 완전)
3. 각 데이터셋에 대해 NMF+TGAN 학습
4. 결과 비교 및 분석
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """명령어 실행 및 결과 출력"""
    print(f"\n🚀 {description}")
    print(f"명령어: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ 성공!")
        if result.stdout:
            print("출력:")
            print(result.stdout)
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

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='멀티오믹스 NMF+TGAN 전체 파이프라인')
    
    # 데이터 경로 설정
    parser.add_argument('--raw_data_dir', type=str, default='/home/dyan/data/data',
                       help='원본 데이터 디렉토리')
    parser.add_argument('--subtype_file', type=str, default='/home/dyan/data/data/brca-2Fpam50.tsv',
                       help='PAM50 서브타입 파일 경로')
    parser.add_argument('--output_dir', type=str, default='./multi_omics_results',
                       help='결과 저장 디렉토리')
    
    # 학습 설정
    parser.add_argument('--num_epochs', type=int, default=50, help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='학습률')
    parser.add_argument('--use_attention', action='store_true', help='Cross-Attention 사용')
    
    # 실행 옵션
    parser.add_argument('--skip_preprocessing', action='store_true', help='전처리 건너뛰기')
    parser.add_argument('--skip_training', action='store_true', help='학습 건너뛰기')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    
    args = parser.parse_args()
    
    print("🎯 멀티오믹스 NMF+TGAN 전체 파이프라인 시작")
    print("=" * 80)
    print(f"원본 데이터 디렉토리: {args.raw_data_dir}")
    print(f"서브타입 파일: {args.subtype_file}")
    print(f"결과 저장 디렉토리: {args.output_dir}")
    print(f"학습 에포크: {args.num_epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"Cross-Attention 사용: {'예' if args.use_attention else '아니오'}")
    print("=" * 80)
    
    # 결과 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1단계: 데이터 전처리
    if not args.skip_preprocessing:
        print("\n📊 1단계: 멀티오믹스 데이터 전처리")
        
        # 전처리 스크립트 실행 (향상된 버전 사용)
        prep_script = "data_prep_enhanced.py"
        if os.path.exists(prep_script):
            prep_command = f"python {prep_script} --rna {args.raw_data_dir}/TCGA-BRCA.star_tpm.tsv --methy {args.raw_data_dir}/TCGA-BRCA.methylation450.tsv --protein {args.raw_data_dir}/TCGA-BRCA.protein.tsv --subtype {args.subtype_file} --out_dir ./processed_datasets --prefix BRCA_PAM50"
            if not run_command(prep_command, "향상된 데이터 전처리 실행"):
                print("❌ 데이터 전처리 실패. 파이프라인을 중단합니다.")
                return False
        else:
            print(f"❌ 전처리 스크립트를 찾을 수 없습니다: {prep_script}")
            return False
    
    # 2단계: 각 데이터셋에 대해 NMF+TGAN 학습
    if not args.skip_training:
        print("\n🤖 2단계: NMF+TGAN 모델 학습")
        
        # 사용 가능한 데이터셋 타입 확인
        processed_dir = "./processed_datasets"
        dataset_types = []
        
        if os.path.exists(processed_dir):
            for dataset_type in ['original', 'noisy', 'complete']:
                dataset_path = os.path.join(processed_dir, dataset_type)
                if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0:
                    dataset_types.append(dataset_type)
        
        if not dataset_types:
            print("❌ 전처리된 데이터셋을 찾을 수 없습니다.")
            return False
        
        print(f"사용 가능한 데이터셋: {dataset_types}")
        
        for dataset_type in dataset_types:
            print(f"\n📚 {dataset_type.upper()} 데이터셋 학습 시작")
            
            # 학습 명령어 구성
            train_command = f"python multi_omics_train.py"
            train_command += f" --dataset_type {dataset_type}"
            train_command += f" --data_dir ./processed_datasets"
            train_command += f" --num_epochs {args.num_epochs}"
            train_command += f" --batch_size {args.batch_size}"
            train_command += f" --learning_rate {args.learning_rate}"
            train_command += f" --gpu_id {args.gpu_id}"
            train_command += f" --save_dir {args.output_dir}/{dataset_type}_models"
            
            if args.use_attention:
                train_command += " --use_attention"
            
            # 학습 실행
            if not run_command(train_command, f"{dataset_type} 데이터셋 학습"):
                print(f"⚠️ {dataset_type} 데이터셋 학습에 실패했습니다. 계속 진행합니다.")
    
    # 3단계: 결과 요약 및 비교
    print("\n📊 3단계: 결과 요약 및 비교")
    
    # 결과 디렉토리 구조 확인
    print("\n📁 생성된 결과 디렉토리:")
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # 처음 5개 파일만 표시
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... ({len(files)}개 파일)")
    
    # 학습 결과 요약
    print("\n🎯 학습 결과 요약:")
    for dataset_type in dataset_types:
        model_dir = f"{args.output_dir}/{dataset_type}_models"
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if model_files:
                # 가장 최근 모델 찾기
                latest_model = sorted(model_files)[-1]
                print(f"  {dataset_type.upper()}: {len(model_files)}개 모델, 최신: {latest_model}")
            else:
                print(f"  {dataset_type.upper()}: 모델 파일 없음")
        else:
            print(f"  {dataset_type.upper()}: 디렉토리 없음")
    
    # 다음 단계 안내
    print("\n🔮 다음 단계:")
    print("1. TensorBoard로 학습 과정 확인:")
    for dataset_type in dataset_types:
        log_dir = f"{args.output_dir}/{dataset_type}_models"
        if os.path.exists(log_dir):
            print(f"   tensorboard --logdir {log_dir}")
    
    print("\n2. 결과 분석 및 시각화:")
    print("   - 각 데이터셋별 성능 비교")
    print("   - Cross-Attention 효과 분석")
    print("   - 멀티오믹스 통합 품질 평가")
    
    print("\n3. 생물학적 검증:")
    print("   - 서브타입별 성능 분석")
    print("   - 생물학적 경로 분석")
    print("   - 임상 예후 예측 성능")
    
    print("\n🎉 파이프라인 완료!")
    print(f"결과는 {args.output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()


