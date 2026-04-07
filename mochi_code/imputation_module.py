#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tri-joint 모델을 사용한 멀티오믹스 데이터 Imputation 모듈

사용법:
    from imputation_module import TriJointImputer
    
    imputer = TriJointImputer(checkpoint_path, device='cuda:0')
    imputer.impute_dataset(rna_data, protein_data, methyl_data, 
                          '10p', output_dir)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class TriJointImputer:
    """
    Tri-joint 모델을 사용한 멀티오믹스 데이터 Imputation 클래스
    
    Attributes:
        device: 사용할 디바이스 (GPU/CPU)
        Gp: RNA+Meth → Protein Generator
        Gr: Prot+Meth → RNA Generator  
        Gm: RNA+Prot → Meth Generator
        batch_size: 배치 크기
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0', batch_size: int = 64):
        """
        Args:
            checkpoint_path: 학습된 모델 체크포인트 경로
            device: 사용할 디바이스 ('cuda:0', 'cpu')
            batch_size: 배치 크기
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        print(f"🎯 디바이스: {self.device}")
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        
        # 모델 생성 및 가중치 로드
        self._load_models(checkpoint_path)
        print("✅ Tri-joint 모델 로드 완료!")
    
    def _load_models(self, checkpoint_path: str):
        """체크포인트에서 모델 로드"""
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Generator 클래스 임포트
        from models import Generator
        
        # Generator 객체 생성 (학습과 동일한 아키텍처)
        self.Gp = Generator(input_size=60660+10000, output_size=487, 
                           use_attn=True, n_heads=4, d_head=64)
        self.Gr = Generator(input_size=487+10000, output_size=60660, 
                           use_attn=True, n_heads=4, d_head=64)
        self.Gm = Generator(input_size=60660+487, output_size=10000, 
                           use_attn=True, n_heads=4, d_head=64)
        
        # GPU로 이동
        self.Gp = self.Gp.to(self.device)
        self.Gr = self.Gr.to(self.device)
        self.Gm = self.Gm.to(self.device)
        
        # 가중치 로드
        self.Gp.load_state_dict(checkpoint['Gp'])
        self.Gr.load_state_dict(checkpoint['Gr'])
        self.Gm.load_state_dict(checkpoint['Gm'])
        
        # 평가 모드로 설정
        self.Gp.eval()
        self.Gr.eval()
        self.Gm.eval()
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        데이터 전처리: 결측값을 0으로 채우고 텐서로 변환 (전치 포함)
        
        Args:
            data: 결측값이 포함된 DataFrame
            
        Returns:
            (filled_tensor, missing_mask): 전치된 텐서와 결측 마스크
        """
        # 결측값 마스크 생성 (전치됨)
        missing_mask = torch.tensor(data.isna().values, dtype=torch.bool, device=self.device).T
        
        # 결측값을 0으로 채우고 텐서로 변환 (전치됨)
        data_filled = data.fillna(0)
        filled_tensor = torch.tensor(data_filled.values, dtype=torch.float32, device=self.device).T
        
        return filled_tensor, missing_mask
    
    def _ensure_same_shape(self, mask: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """마스크와 텐서의 shape을 맞춤"""
        if mask.shape != tensor.shape:
            if mask.T.shape == tensor.shape:
                mask = mask.T
            else:
                raise ValueError(f"마스크 shape {mask.shape} != 텐서 shape {tensor.shape}")
        return mask
    
    def _batch_forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """배치 단위로 모델 추론"""
        model.eval()
        with torch.no_grad():
            if self.batch_size is None:
                return model(x)
            
            outputs = []
            for i in range(0, x.shape[0], self.batch_size):
                batch_x = x[i:i+self.batch_size]
                batch_output = model(batch_x)
                outputs.append(batch_output)
            
            return torch.cat(outputs, dim=0)
    
    def _tensor_to_dataframe(self, tensor: torch.Tensor, 
                            reference_df: pd.DataFrame) -> pd.DataFrame:
        """텐서를 DataFrame으로 변환 (원본 구조 유지)"""
        arr = tensor.detach().cpu().numpy()
        
        # shape 확인 및 조정
        if arr.shape == (len(reference_df.columns), len(reference_df.index)):
            # (samples, features) → (features, samples)로 전치
            arr = arr.T
        elif arr.shape != (len(reference_df.index), len(reference_df.columns)):
            raise ValueError(f"예상치 못한 shape: {arr.shape}")
        
        return pd.DataFrame(arr, index=reference_df.index, columns=reference_df.columns)
    
    def impute_dataset(self, rna_data: pd.DataFrame, protein_data: pd.DataFrame, 
                      methyl_data: pd.DataFrame, missing_rate_tag: str, 
                      output_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """
        데이터셋 imputation 수행
        
        Args:
            rna_data: RNA 데이터 (결측값 포함)
            protein_data: Protein 데이터 (결측값 포함)
            methyl_data: Methylation 데이터 (결측값 포함)
            missing_rate_tag: 결측 비율 태그 (예: '10p', '30p', '50p')
            output_dir: 결과 저장 디렉토리
            
        Returns:
            imputed_data: imputation된 데이터 딕셔너리
        """
        print(f"\n🔧 {missing_rate_tag} 결측 데이터 imputation 시작...")
        
        # 출력 디렉토리 생성
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 전처리
        rna_tensor, rna_mask = self._prepare_data(rna_data)
        protein_tensor, protein_mask = self._prepare_data(protein_data)
        methyl_tensor, methyl_mask = self._prepare_data(methyl_data)
        
        # 마스크 shape 맞춤
        rna_mask = self._ensure_same_shape(rna_mask, rna_tensor)
        protein_mask = self._ensure_same_shape(protein_mask, protein_tensor)
        methyl_mask = self._ensure_same_shape(methyl_mask, methyl_tensor)
        
        # Tri-joint 모델로 imputation 수행
        print("  🚀 Tri-joint 모델로 imputation 수행 중...")
        
        # 입력 결합 (학습과 동일한 순서)
        # 이미 전치된 텐서: (N, features)
        X_Gp = torch.cat([rna_tensor, methyl_tensor], dim=1)  # RNA+Meth → Protein
        X_Gr = torch.cat([protein_tensor, methyl_tensor], dim=1)  # Prot+Meth → RNA
        X_Gm = torch.cat([rna_tensor, protein_tensor], dim=1)  # RNA+Prot → Meth
        
        # 배치 추론
        protein_pred = self._batch_forward(self.Gp, X_Gp)
        rna_pred = self._batch_forward(self.Gr, X_Gr)
        methyl_pred = self._batch_forward(self.Gm, X_Gm)
        
        # 관측은 유지, 결측만 예측값으로 치환
        # 이미 전치된 텐서와 마스크
        protein_imputed = torch.where(protein_mask, protein_pred, protein_tensor)
        rna_imputed = torch.where(rna_mask, rna_pred, rna_tensor)
        methyl_imputed = torch.where(methyl_mask, methyl_pred, methyl_tensor)
        
        # DataFrame으로 변환
        rna_imputed_df = self._tensor_to_dataframe(rna_imputed, rna_data)
        protein_imputed_df = self._tensor_to_dataframe(protein_imputed, protein_data)
        methyl_imputed_df = self._tensor_to_dataframe(methyl_imputed, methyl_data)
        
        # 결과 저장
        rna_imputed_df.to_csv(output_dir / f"rna_full_{missing_rate_tag}.tsv", sep='\t')
        protein_imputed_df.to_csv(output_dir / f"protein_full_{missing_rate_tag}.tsv", sep='\t')
        methyl_imputed_df.to_csv(output_dir / f"methyl_full_{missing_rate_tag}.tsv", sep='\t')
        
        print(f"  ✅ {missing_rate_tag} imputation 완료!")
        print(f"  💾 저장 위치: {output_dir}")
        
        # 결과 반환
        imputed_data = {
            'rna': rna_imputed_df,
            'protein': protein_imputed_df,
            'methyl': methyl_imputed_df
        }
        
        return imputed_data
    
    def impute_multiple_datasets(self, datasets: Dict[str, Dict[str, pd.DataFrame]], 
                                output_base_dir: Union[str, Path]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        여러 데이터셋을 한 번에 imputation
        
        Args:
            datasets: 데이터셋 딕셔너리
                {
                    '10p': {'rna': df, 'protein': df, 'methyl': df},
                    '30p': {'rna': df, 'protein': df, 'methyl': df},
                    '50p': {'rna': df, 'protein': df, 'methyl': df}
                }
            output_base_dir: 기본 출력 디렉토리
            
        Returns:
            모든 imputation 결과
        """
        output_base_dir = Path(output_base_dir)
        all_results = {}
        
        for tag, data_dict in datasets.items():
            output_dir = output_base_dir / f"imputation_miss{tag}"
            
            result = self.impute_dataset(
                rna_data=data_dict['rna'],
                protein_data=data_dict['protein'],
                methyl_data=data_dict['methyl'],
                missing_rate_tag=tag,
                output_dir=output_dir
            )
            
            all_results[tag] = result
        
        return all_results


def main():
    """테스트용 메인 함수"""
    print("🧪 TriJointImputer 테스트")
    
    # 체크포인트 경로
    checkpoint_path = "/home/dyan/nmf/mochi_code/results/tri_joint_v2/tri_best.ckpt"
    
    # Imputer 생성
    imputer = TriJointImputer(checkpoint_path, device='cuda:0')
    
    print("✅ Imputer 생성 완료!")


if __name__ == "__main__":
    main()
