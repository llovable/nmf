#!/usr/bin/env python3
"""
멀티오믹스 데이터를 위한 NMF+TGAN 학습 스크립트
- 3개 오믹스 데이터 간 상호 결측 보정
- RNA, DNA methylation, Protein 데이터 통합
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

from models import Generator, Critic
from utils import nmf_U_V
import sys
sys.path.append('.')
from dataloader import get_multi_omics_dataloaders, get_multi_omics_dataloaders_concat, get_triple_dataloaders
import torch.linalg as LA

# 🌱 시드 고정 유틸
def set_seed(seed):
    if seed is None: 
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiOmicsTrainer:
    def __init__(self, config):
        """
        멀티오믹스 NMF+TGAN 학습기
        
        Args:
            config: 학습 설정 딕셔너리
        """
        self.config = config
        self.device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
        
        # 데이터 경로
        self.data_dir = config.get('data_dir', './processed_datasets')
        self.dataset_type = config.get('dataset_type', 'original')  # original, noise, complete
        
        # 모델 설정
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.00005)
        self.num_epochs = config.get('num_epochs', 100)
        self.n_critic = config.get('n_critic', 5)
        self.clip_value = config.get('clip_value', 0.01)
        
        # 손실 가중치
        self.mse_weight = config.get('mse_weight', 0.1)
        self.nmf_weight = config.get('nmf_weight', 0.1)
        self.gen_weight = config.get('gen_weight', 1.0)
        
        # NMF 설정
        self.nmf_k = config.get('nmf_k', 20)
        self.ridge_alpha = config.get('ridge_alpha', 1e-3)
        
        # 동적 마스킹 설정
        self.dynamic_masking = config.get('dynamic_masking', False)
        self.mask_rate_range = config.get('mask_rate_range', (0.05, 0.5))
        
        # 🚀 Early Stopping 설정
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.min_epochs = config.get('min_epochs', 5)  # 최소 에포크 수
        
        # 모델 초기화
        self.models = {}
        self.optimizers = {}
        self.dataloaders = {}
        
        # 결과 저장
        self.save_dir = config.get('save_dir', f'./saved_models/{self.dataset_type}')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.save_dir)
        
        # 🕐 Early Stopping 추적
        self.best_val_mse = float('inf')
        self.patience_counter = 0
        
    def _validate_batch_size(self, num_samples, batch_size):
        """배치 크기 검증 및 자동 조정"""
        if batch_size > num_samples:
            new_batch_size = max(1, num_samples // 2)
            print(f"⚠️ 배치 크기 {batch_size}가 샘플 수 {num_samples}보다 큽니다.")
            print(f"   자동으로 {new_batch_size}로 조정합니다.")
            return new_batch_size
        
        # 마지막 배치가 1개가 되지 않도록 조정
        if num_samples % batch_size == 1:
            new_batch_size = batch_size - 1
            if new_batch_size >= 2:
                print(f"⚠️ 마지막 배치가 1개가 됩니다. 배치 크기를 {new_batch_size}로 조정합니다.")
                return new_batch_size
        
        return batch_size

    def load_data(self):
        """멀티오믹스 데이터 로드 (플랫 파일 구조 지원)"""
        print(f"📊 {self.dataset_type} 데이터셋 로딩 중...")
        print(f"🔍 dataset_type: {self.dataset_type}")
        print(f"🔍 data_dir: {self.data_dir}")
        
        # src_combo 설정 확인
        self.src_combo = self.config.get('src_combo', None)
        if self.src_combo:
            print(f"🔍 src_combo: {self.src_combo}")
        
        # 플랫 파일 구조에서 데이터 로드
        self.omics_data = {}
        
        # 파일 패턴(평탄화): BRCA_PAM50.{omics}.{dataset_type}.tsv
        for omics in ['rna', 'methy', 'protein']:
            pattern = f"BRCA_PAM50.{omics}.{self.dataset_type}.tsv"
            
            print(f"🔍 {omics} 패턴: {pattern}")
            
            file_path = os.path.join(self.data_dir, pattern)
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, sep='\t', index_col=0)
                # omics 키를 일관성 있게 유지 (methy -> meth)
                omics_key = 'meth' if omics == 'methy' else omics
                self.omics_data[omics_key] = df
                print(f"  {omics_key.upper()}: {df.shape}")
            else:
                print(f"❌ {omics} 데이터 파일을 찾을 수 없습니다: {file_path}")
                return False
        
        # 공통 샘플 정보 생성
        common_samples = set.intersection(*[set(df.columns) for df in self.omics_data.values()])
        self.samples_info = pd.DataFrame({
            'sample_id': sorted(list(common_samples)),
            'subtype': ['Unknown'] * len(common_samples)
        })
        print(f"  공통 샘플: {len(common_samples)}개")
        
        return True
    
    def prepare_dataloaders(self):
        """데이터로더 준비"""
        print("🔄 데이터로더 준비 중...")
        
        # 공통 샘플 ID 추출
        common_samples = self.samples_info['sample_id'].tolist()
        
        # 각 오믹스 데이터에서 공통 샘플만 선택
        for omics, df in self.omics_data.items():
            available_samples = [s for s in common_samples if s in df.columns]
            self.omics_data[omics] = df[available_samples]
            print(f"  {omics.upper()} 사용 가능 샘플: {len(available_samples)}")
        
        # 최종 공통 샘플 (모든 오믹스에 존재하는 샘플)
        final_common = set.intersection(*[set(df.columns) for df in self.omics_data.values()])
        final_common = sorted(list(final_common))
        print(f"최종 공통 샘플: {len(final_common)}개")
        
        # 데이터 정렬 및 전처리
        for omics in self.omics_data:
            self.omics_data[omics] = self.omics_data[omics][final_common]
        
        # src_combo가 설정된 경우 concat 데이터로더 사용
        if self.src_combo:
            print(f"🔄 src_combo 모드: {self.src_combo}")
            # src_combo 파싱
            src_modalities = self.src_combo.split("+")
            tgt_modality = self.config.get('tgt_modality', 'protein')
            
            print(f"  소스 모달리티: {src_modalities}")
            print(f"  타깃 모달리티: {tgt_modality}")
            
            # concat 데이터로더 생성
            try:
                from dataloader import get_multi_omics_dataloaders_concat
                
                # 데이터 테이블 준비
                tables = {}
                for m in src_modalities + [tgt_modality]:
                    if m == 'methy':
                        m_key = 'meth'
                    elif m == 'methyl':
                        m_key = 'meth'
                    else:
                        m_key = m
                    
                    if m_key in self.omics_data:
                        tables[m] = self.omics_data[m_key].T  # [samples, features]로 변환
                    else:
                        print(f"❌ 모달리티 {m} ({m_key})를 찾을 수 없습니다")
                        return False
                
                # concat 데이터로더 생성
                train_loader, valid_loader, _, data_info = get_multi_omics_dataloaders_concat(
                    data_dir=self.data_dir,
                    src_combo=src_modalities,
                    tgt_modality=tgt_modality,
                    prefix="BRCA_PAM50",
                    batch_size=self.batch_size
                )
                
                self.dataloaders = {
                    'train': train_loader,
                    'val': valid_loader
                }
                
                print(f"✅ concat 데이터로더 생성 완료")
                print(f"  소스 차원: {data_info['src_dim']}")
                print(f"  타깃 차원: {data_info['tgt_dim']}")
                print(f"  훈련 샘플: {data_info['n_train']}")
                print(f"  검증 샘플: {data_info['n_valid']}")
                
                return True
                
            except Exception as e:
                print(f"❌ concat 데이터로더 생성 실패: {e}")
                return False
        
        # 기존 방식: 각 오믹스별 개별 데이터로더
        print(f"🔄 기존 방식: 각 오믹스별 개별 데이터로더")
        
        # 학습/검증/테스트 분할
        np.random.seed(42)
        np.random.shuffle(final_common)
        
        train_size = int(0.7 * len(final_common))
        val_size = int(0.15 * len(final_common))
        
        train_samples = final_common[:train_size]
        val_samples = final_common[train_size:train_size + val_size]
        test_samples = final_common[train_size + val_size:]
        
        print(f"  학습: {len(train_samples)}, 검증: {len(val_samples)}, 테스트: {len(test_samples)}")
        
        # 배치 크기 검증 및 자동 조정
        self.batch_size = self._validate_batch_size(len(train_samples), self.batch_size)
        print(f"✅ 최종 배치 크기: {self.batch_size}")
        
        # 데이터로더 생성
        self.dataloaders = {
            'train': self._create_dataloader(train_samples),
            'val': self._create_dataloader(val_samples),
            'test': self._create_dataloader(test_samples)
        }
        
        return True
    
    def _create_dataloader(self, samples):
        """특정 샘플들에 대한 데이터로더 생성"""
        data_list = []
        
        for sample in samples:
            # 각 오믹스에서 해당 샘플의 데이터 추출
            sample_data = {}
            for omics, df in self.omics_data.items():
                values = df[sample].values
                # NaN을 0으로 치환하고 마스크 생성
                mask = ~np.isnan(values)
                values_clean = np.nan_to_num(values, nan=0.0)
                
                sample_data[omics] = {
                    'values': values_clean.astype(np.float32),
                    'mask': mask.astype(np.float32)
                }
            
            data_list.append(sample_data)
        
        return data_list
    
    def initialize_models(self):
        """모델 초기화"""
        print("🤖 모델 초기화 중...")
        
        # 각 오믹스별 Generator 생성
        for omics in self.omics_data:
            input_size = self.omics_data[omics].shape[0]  # 특징 수
            
            # 다른 오믹스들을 타겟으로 하는 Generator들
            for target_omics in self.omics_data:
                if target_omics != omics:
                    model_key = f"gen_{omics}_to_{target_omics}"
                    
                    # ★ 타깃 오믹스에 따라 nonneg_output 자동 설정
                    nonneg_output = target_omics in ['rna', 'meth']  # RNA/methylation은 비음수
                    
                    self.models[model_key] = Generator(
                        input_size=input_size,
                        output_size=self.omics_data[target_omics].shape[0],
                        use_attn=self.config.get('use_attention', False),
                        nonneg_output=nonneg_output,  # 자동 설정
                        n_src_tokens=self.config.get('n_src_tokens', 8)  # 소스 토큰 수
                    ).to(self.device)
                    
                    # 옵티마이저
                    self.optimizers[f"opt_{model_key}"] = optim.RMSprop(
                        self.models[model_key].parameters(),
                        lr=self.learning_rate
                    )
        
        # 각 오믹스별 Critic 생성
        for omics in self.omics_data:
            model_key = f"critic_{omics}"
            self.models[model_key] = Critic(
                input_size=self.omics_data[omics].shape[0]
            ).to(self.device)
            
            # 옵티마이저
            self.optimizers[f"opt_{model_key}"] = optim.RMSprop(
                self.models[model_key].parameters(),
                lr=self.learning_rate
            )
        
        print(f"  생성된 모델: {len([k for k in self.models.keys() if k.startswith('gen')])}개 Generator")
        print(f"  생성된 Critic: {len([k for k in self.models.keys() if k.startswith('critic')])}개")
        
        return True
    
    def _compute_nmf_loss(self, fake_data, target_omics):
        """
        fake_data: [B, features]  (우리 코드에선 B=1 형태)
        NMF는 비음수를 요구 → basis 만들 때 적용한 변환을 동일하게 적용
        """
        if target_omics not in self.nmf_basis:
            return torch.tensor(0.0, device=self.device)

        basis = self.nmf_basis[target_omics]
        D = basis["D"].to(self.device)
        DtD_inv = basis["DtD_inv"].to(self.device)
        shift = basis["shift"].to(self.device)  # [features]

        # fake를 NMF 좌표계로 사상
        if target_omics in ["rna", "meth"]:
            Y = torch.clamp(fake_data, min=0.0)  # 비음수 클립
        else:  # protein
            Y = fake_data + shift                # 행별 시프트 적용
            Y = torch.clamp(Y, min=1e-8)         # 비음수 보장

        # Û = Y·D·(DᵀD+αI)⁻¹  ,  Y ≈ Û·Dᵀ
        U_hat = (Y @ D) @ DtD_inv
        Y_rec = U_hat @ D.T
        nmf_loss = nn.functional.mse_loss(Y_rec, Y)
        return nmf_loss
    
    def _make_nonnegative_with_shift(self, X: np.ndarray, modality: str):
        """
        NMF 입력용 비음수화.
        - RNA/meth: 음수는 0으로 클립, shift=0
        - protein : 행별 최소값이 음수면 그 절댓값만큼 올려서 ≥0, shift=per-row
        반환: (X_nonneg, shift)  # shift shape = [n_features, 1]
        """
        X = np.nan_to_num(X, nan=0.0)
        if modality in ["rna", "meth"]:
            X_nonneg = np.where(X < 0, 0.0, X)
            shift = np.zeros((X.shape[0], 1), dtype=X.dtype)
        else:  # protein
            min_row = X.min(axis=1, keepdims=True)           # [n_features, 1]
            shift = np.where(min_row < 0, -min_row, 0.0)     # ≥0
            X_nonneg = X + shift + 1e-8                      # 완전 0 회피
        return X_nonneg, shift
    
    def prepare_nmf_basis(self):
        """NMF 기저 행렬 준비 (가속용)"""
        print("🔧 NMF 기저 행렬 준비 중...")
        
        self.nmf_basis = {}
        
        for omics in self.omics_data:
            X = self.omics_data[omics].values  # [features, samples]
            X_nonneg, shift = self._make_nonnegative_with_shift(X, modality=omics)

            U, V = nmf_U_V(X_nonneg.T, k=self.nmf_k)  # sklearn NMF

            D = torch.tensor(V.T, dtype=torch.float32)         # [features, k]
            DtD = (D.T @ D)
            DtD_inv = LA.inv(DtD + self.ridge_alpha * torch.eye(DtD.shape[0]))
            shift_t = torch.tensor(shift.squeeze(), dtype=torch.float32)  # [features]

            self.nmf_basis[omics] = {
                "D": D, "DtD_inv": DtD_inv, "U_global": U, "shift": shift_t
            }
            print(f"  {omics.upper()}: D {D.shape}, min(after nonneg)={X_nonneg.min():.4f}")
    
    def train_epoch(self, epoch):
        """한 에포크 학습"""
        # 학습 모드
        for model in self.models.values():
            model.train()
        
        total_losses = {
            'gen_loss': 0,
            'critic_loss': 0,
            'mse_loss': 0,
            'nmf_loss': 0
        }
        
        num_batches = len(self.dataloaders['train'])
        
        for batch_idx, batch_data in enumerate(tqdm(self.dataloaders['train'], desc=f"Epoch {epoch}")):
            batch_losses = self._train_batch(batch_data)
            
            # 손실 누적
            for key in total_losses:
                total_losses[key] += batch_losses[key]
        
        # 평균 손실 계산
        for key in total_losses:
            total_losses[key] /= num_batches
        
        # TensorBoard에 기록
        for key, value in total_losses.items():
            self.writer.add_scalar(f"Loss/{key}", value, epoch)
        
        return total_losses
    
    def _train_batch(self, batch_data):
        """배치 데이터 학습"""
        batch_losses = {
            'gen_loss': 0,
            'critic_loss': 0,
            'mse_loss': 0,
            'nmf_loss': 0
        }
        
        # 각 오믹스 쌍에 대해 학습
        for source_omics in self.omics_data:
            for target_omics in self.omics_data:
                if source_omics == target_omics:
                    continue
                
                # Generator와 Critic 키
                gen_key = f"gen_{source_omics}_to_{target_omics}"
                critic_key = f"critic_{target_omics}"
                
                if gen_key not in self.models:
                    continue
                
                # 데이터 준비
                source_values = torch.tensor(
                    batch_data[source_omics]['values'], 
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                target_values = torch.tensor(
                    batch_data[target_omics]['values'], 
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                target_mask = torch.tensor(
                    batch_data[target_omics]['mask'], 
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                # Critic 학습
                for _ in range(self.n_critic):
                    self.optimizers[f"opt_{critic_key}"].zero_grad()
                    
                    # 진짜 데이터
                    real_output = self.models[critic_key](target_values)
                    
                    # 가짜 데이터
                    fake_data = self.models[gen_key](source_values)
                    fake_output = self.models[critic_key](fake_data.detach())
                    
                    # WGAN 손실
                    critic_loss = -torch.mean(real_output) + torch.mean(fake_output)
                    critic_loss.backward()
                    self.optimizers[f"opt_{critic_key}"].step()
                    
                    # 파라미터 클리핑
                    for p in self.models[critic_key].parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)
                    
                    batch_losses['critic_loss'] += critic_loss.item()
                
                # Generator 학습
                self.optimizers[f"opt_{gen_key}"].zero_grad()
                
                fake_data = self.models[gen_key](source_values)
                fake_output = self.models[critic_key](fake_data)
                
                # Generator 손실
                gen_loss = -torch.mean(fake_output)
                
                # MSE 손실 (동적 마스킹 또는 배치 마스크 적용)
                if self.dynamic_masking:
                    # 동적 마스킹: 배치마다 pseudo-missing 샘플링
                    import numpy as np
                    rate = float(np.random.uniform(self.mask_rate_range[0], self.mask_rate_range[1]))
                    dyn_mask = (torch.rand_like(target_values) < rate).float() * (1.0 - target_mask)
                    mse_mask = dyn_mask
                else:
                    # 기존 방식: 배치의 마스크 사용
                    mse_mask = target_mask
                
                if mse_mask.sum() > 0:
                    mse_loss = torch.mean(((target_values - fake_data) ** 2) * mse_mask)
                else:
                    mse_loss = torch.tensor(0.0, device=self.device)
                
                # NMF 손실 (가속)
                nmf_loss = self._compute_nmf_loss(fake_data, target_omics)
                
                # 총 손실
                total_gen_loss = (
                    gen_loss * self.gen_weight +
                    mse_loss * self.mse_weight +
                    nmf_loss * self.nmf_weight
                )
                
                total_gen_loss.backward()
                self.optimizers[f"opt_{gen_key}"].step()
                
                # 손실 기록
                batch_losses['gen_loss'] += gen_loss.item()
                batch_losses['mse_loss'] += mse_loss.item()
                batch_losses['nmf_loss'] += nmf_loss.item()
        
        return batch_losses
    
    # ⚠️ (중복 정의 제거됨) 위의 _compute_nmf_loss 하나만 사용합니다.
    
    def validate(self, epoch):
        """검증"""
        # 평가 모드
        for model in self.models.values():
            model.eval()
        
        total_mse = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch_data in self.dataloaders['val']:
                for source_omics in self.omics_data:
                    for target_omics in self.omics_data:
                        if source_omics == target_omics:
                            continue
                        
                        gen_key = f"gen_{source_omics}_to_{target_omics}"
                        if gen_key not in self.models:
                            continue
                        
                        # 데이터 준비
                        source_values = torch.tensor(
                            batch_data[source_omics]['values'], 
                            dtype=torch.float32
                        ).unsqueeze(0).to(self.device)
                        
                        target_values = torch.tensor(
                            batch_data[target_omics]['values'], 
                            dtype=torch.float32
                        ).unsqueeze(0).to(self.device)
                        
                        target_mask = torch.tensor(
                            batch_data[target_omics]['mask'], 
                            dtype=torch.float32
                        ).unsqueeze(0).to(self.device)
                        
                        # 예측
                        fake_data = self.models[gen_key](source_values)
                        
                        # 마스크된 MSE
                        mse = torch.mean(((target_values - fake_data) ** 2) * target_mask)
                        total_mse += mse.item()
                        num_samples += 1
        
        if num_samples > 0:
            avg_mse = total_mse / num_samples
            self.writer.add_scalar("Validation/MSE", avg_mse, epoch)
            return avg_mse
        
        return float('inf')
    
    def save_models(self, epoch, val_mse):
        """모델 저장"""
        save_path = os.path.join(self.save_dir, f"epoch_{epoch}_mse_{val_mse:.6f}.pth")
        
        save_dict = {
            'epoch': epoch,
            'val_mse': val_mse,
            'config': self.config,
            'models': {},
            'optimizers': {}
        }
        
        # 모델 상태 저장
        for name, model in self.models.items():
            save_dict['models'][name] = model.state_dict()
        
        # 옵티마이저 상태 저장
        for name, optimizer in self.optimizers.items():
            save_dict['optimizers'][name] = optimizer.state_dict()
        
        torch.save(save_dict, save_path)
        print(f"💾 모델 저장됨: {save_path}")
    
    def train(self):
        """전체 학습 과정"""
        print("🚀 멀티오믹스 NMF+TGAN 학습 시작")
        print("=" * 60)
        
        # 데이터 로드
        if not self.load_data():
            print("❌ 데이터 로드 실패")
            return
        
        # 데이터로더 준비
        if not self.prepare_dataloaders():
            print("❌ 데이터로더 준비 실패")
            return
        
        # 모델 초기화
        if not self.initialize_models():
            print("❌ 모델 초기화 실패")
            return
        
        # NMF 기저 준비
        self.prepare_nmf_basis()
        
        # 학습 루프
        print(f"🚀 Early Stopping 설정: {self.early_stopping_patience} 에포크 동안 개선 없으면 중단")
        print(f"📚 최소 학습 에포크: {self.min_epochs}")
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n📚 Epoch {epoch}/{self.num_epochs}")
            
            # 학습
            train_losses = self.train_epoch(epoch)
            
            # 검증
            val_mse = self.validate(epoch)
            
            # 모델 저장 (최고 성능)
            if val_mse < self.best_val_mse:
                self.best_val_mse = val_mse
                self.patience_counter = 0  # 개선됨
                self.save_models(epoch, val_mse)
                print(f"  🎯 새로운 최고 성능! 검증 MSE: {val_mse:.6f}")
            else:
                self.patience_counter += 1  # 개선되지 않음
                print(f"  ⏳ {self.patience_counter}/{self.early_stopping_patience} 에포크 동안 개선 없음")
            
            # 진행 상황 출력
            print(f"  학습 손실: {train_losses}")
            print(f"  검증 MSE: {val_mse:.6f}")
            print(f"  최고 검증 MSE: {self.best_val_mse:.6f}")
            
            # 🚀 Early Stopping 체크
            if epoch >= self.min_epochs and self.patience_counter >= self.early_stopping_patience:
                print(f"\n🛑 Early Stopping! {self.early_stopping_patience} 에포크 동안 개선되지 않음")
                print(f"🎯 최종 모델 저장됨 (Epoch {epoch})")
                break
        
        print("\n🎉 학습 완료!")
        print(f"📊 최종 성능: 검증 MSE = {self.best_val_mse:.6f}")
        self.writer.close()


# ========================================
# Tri-joint 트레이너 (3개 오믹스 동시 학습)
# ========================================

class WGAN_GP_Trainer_Tri:
    """
    한 번의 학습 러닝에서 세 타깃을 동시 최적화:
      - Gp: [rna+methy]  -> protein
      - Gr: [prot+methy] -> rna
      - Gm: [rna+prot]   -> methyl
    """
    def __init__(self, Gp, Gr, Gm, Dp, Dr, Dm, device,
                 lr_g=1e-4, lr_d=1e-4, lambda_gp=10.0, n_critic=5,
                 dynamic_masking: bool=False, mask_rate_range=(0.05, 0.5)):
        self.Gp, self.Gr, self.Gm = Gp.to(device), Gr.to(device), Gm.to(device)
        self.Dp, self.Dr, self.Dm = Dp.to(device), Dr.to(device), Dm.to(device)
        self.device = device
        self.lambda_gp, self.n_critic = lambda_gp, n_critic
        self.dynamic_masking, self.mask_rate_range = dynamic_masking, mask_rate_range

        self.optimizer_g = optim.Adam(
            list(Gp.parameters())+list(Gr.parameters())+list(Gm.parameters()),
            lr=lr_g, betas=(0.5,0.9))
        self.optimizer_d = optim.Adam(
            list(Dp.parameters())+list(Dr.parameters())+list(Dm.parameters()),
            lr=lr_d, betas=(0.5,0.9))

    def _gp(self, D, real, fake):
        alpha = torch.rand(real.size(0), 1, device=self.device).expand_as(real)
        inter = (alpha*real + (1-alpha)*fake).requires_grad_(True)
        out = D(inter)
        grad = torch.autograd.grad(out, inter, grad_outputs=torch.ones_like(out),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def _dyn_mask(self, real_mask, like):
        # real_mask: 1=실제 NA → 관측 가능한 위치에서만 pseudo-missing 샘플
        observed = (1.0 - real_mask).clamp(0,1)
        rate = float(np.random.uniform(self.mask_rate_range[0], self.mask_rate_range[1]))
        return (torch.rand_like(like) < rate).float() * observed

    def train_step(self, batch):
        rna   = batch["x_rna"].to(self.device)
        prot  = batch["x_prot"].to(self.device)
        methy = batch["x_methy"].to(self.device)
        m_rna   = batch["m_rna"].to(self.device)
        m_prot  = batch["m_prot"].to(self.device)
        m_methy = batch["m_methy"].to(self.device)

        # --------- Critic ---------
        for _ in range(self.n_critic):
            self.optimizer_d.zero_grad()
            # Protein
            y_p_fake = self.Gp(torch.cat([rna, methy], dim=1), src=None).detach()
            gp = self._gp(self.Dp, prot, y_p_fake)
            loss_Dp = -(self.Dp(prot).mean() - self.Dp(y_p_fake).mean()) + self.lambda_gp*gp
            # RNA
            y_r_fake = self.Gr(torch.cat([prot, methy], dim=1), src=None).detach()
            gr = self._gp(self.Dr, rna, y_r_fake)
            loss_Dr = -(self.Dr(rna).mean() - self.Dr(y_r_fake).mean()) + self.lambda_gp*gr
            # Methyl
            y_m_fake = self.Gm(torch.cat([rna, prot], dim=1), src=None).detach()
            gm = self._gp(self.Dm, methy, y_m_fake)
            loss_Dm = -(self.Dm(methy).mean() - self.Dm(y_m_fake).mean()) + self.lambda_gp*gm
            (loss_Dp + loss_Dr + loss_Dm).backward()
            self.optimizer_d.step()

        # --------- Generator ---------
        self.optimizer_g.zero_grad()
        y_p = self.Gp(torch.cat([rna, methy], dim=1), src=None)
        y_r = self.Gr(torch.cat([prot, methy], dim=1), src=None)
        y_m = self.Gm(torch.cat([rna, prot], dim=1),  src=None)
        g_wgan = - (self.Dp(y_p).mean() + self.Dr(y_r).mean() + self.Dm(y_m).mean())
        # 동적/정적 마스크 기반 MSE
        if self.dynamic_masking:
            mp = self._dyn_mask(m_prot,  y_p)
            mr = self._dyn_mask(m_rna,   y_r)
            mm = self._dyn_mask(m_methy, y_m)
        else:
            mp, mr, mm = m_prot, m_rna, m_methy
        mse = 0.0
        if mp.sum() > 0: mse += ((y_p - prot)*mp).pow(2).mean()
        if mr.sum() > 0: mse += ((y_r - rna)*mr).pow(2).mean()
        if mm.sum() > 0: mse += ((y_m - methy)*mm).pow(2).mean()
        g_total = g_wgan + 0.1*mse
        g_total.backward()
        self.optimizer_g.step()
        return {"D": float((loss_Dp+loss_Dr+loss_Dm).item()), "G": float(g_total.item())}

    def train_epoch(self, loader):
        self.Gp.train(); self.Gr.train(); self.Gm.train()
        self.Dp.train(); self.Dr.train(); self.Dm.train()
        logsD, logsG = [], []
        for batch in loader:
            out = self.train_step(batch)
            logsD.append(out["D"]); logsG.append(out["G"])
        return {"D": float(np.mean(logsD)), "G": float(np.mean(logsG))}

    @torch.no_grad()
    def validate_rmse(self, valid_loader, seed: int = 999, mask_rate: float = 0.30):
        """
        고정 시드/비율로 pseudo-missing을 샘플링하여 모달리티별 RMSE 측정
        """
        rng = np.random.default_rng(seed)
        def fixed_mask(real_mask):
            # real_mask: 1=실제 NA → 관측 위치에서만 샘플링
            observed = (1.0 - real_mask).clamp(0,1)
            samp = torch.from_numpy((rng.random(observed.shape) < mask_rate).astype(np.float32)).to(observed.device)
            return samp * observed

        self.Gp.eval(); self.Gr.eval(); self.Gm.eval()
        sum_sq = {"protein":0.0, "rna":0.0, "methyl":0.0}
        cnt    = {"protein":0.0, "rna":0.0, "methyl":0.0}
        for batch in valid_loader:
            rna   = batch["x_rna"].to(self.device)
            prot  = batch["x_prot"].to(self.device)
            methy = batch["x_methy"].to(self.device)
            m_rna   = batch["m_rna"].to(self.device)
            m_prot  = batch["m_prot"].to(self.device)
            m_methy = batch["m_methy"].to(self.device)
            mp, mr, mm = fixed_mask(m_prot), fixed_mask(m_rna), fixed_mask(m_methy)
            yp = self.Gp(torch.cat([rna, methy], dim=1), src=None)
            yr = self.Gr(torch.cat([prot, methy], dim=1), src=None)
            ym = self.Gm(torch.cat([rna, prot], dim=1),  src=None)
            sum_sq["protein"] += ((prot - yp)*mp).pow(2).sum().item();   cnt["protein"] += mp.sum().item()
            sum_sq["rna"]     += ((rna  - yr)*mr).pow(2).sum().item();   cnt["rna"]     += mr.sum().item()
            sum_sq["methyl"]  += ((methy- ym)*mm).pow(2).sum().item();   cnt["methyl"]  += mm.sum().item()
        rmse = {k: float(np.sqrt(sum_sq[k] / max(cnt[k],1.0))) for k in sum_sq}
        rmse["avg"] = float(np.mean([rmse["protein"], rmse["rna"], rmse["methyl"]]))
        return rmse


# ========================================
# 데이터셋별 자동 학습 함수들
# ========================================

def train_all_variants(
    base_data_dir: str = "/home/dyan/nmf/mochi_code/process_data",
    model_config: dict = None,
    output_base_dir: str = "results",
    # 추가 파라미터들
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    early_stopping_patience: int = 8,
    min_epochs: int = 5,
    gpu_id: int = 0,
    n_critic: int = 5,
    clip_value: float = 0.01,
    mse_weight: float = 0.1,
    nmf_weight: float = 0.1,
    gen_weight: float = 1.0,
    nmf_k: int = 20,
    ridge_alpha: float = 1e-3,
    use_attention: bool = False,
    # 동적 마스킹 옵션 추가
    dynamic_masking: bool = False,
    mask_rate_range: tuple = (0.05, 0.5),
    # src_combo 옵션 추가
    src_combo: str = None,
    tgt_modality: str = 'protein'
):
    """
    모든 결측치 비율 데이터셋에 대해 모델 학습
    
    Args:
        base_data_dir: 전처리된 데이터가 있는 기본 디렉토리
        model_config: 모델 설정 (하이퍼파라미터 등) - None이면 기본값 사용
        output_base_dir: 결과 저장 기본 디렉토리
        
        # 추가 파라미터들
        num_epochs: 최대 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
        early_stopping_patience: Early Stopping 인내심 (에포크 수)
        min_epochs: 최소 학습 에포크 수
        gpu_id: GPU ID
        n_critic: Critic 학습 횟수
        clip_value: WGAN 클리핑 값
        mse_weight: MSE 손실 가중치
        nmf_weight: NMF 손실 가중치
        gen_weight: Generator 손실 가중치
        nmf_k: NMF 클러스터 수
        ridge_alpha: Ridge 정규화 계수
        use_attention: Cross-Attention 사용 여부
    
    Returns:
        dict: 각 데이터셋별 학습 결과
    """
    
    # 기본 모델 설정 (사용자 지정 파라미터로 덮어쓰기)
    if model_config is None:
        model_config = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'n_critic': n_critic,
            'clip_value': clip_value,
            'mse_weight': mse_weight,
            'nmf_weight': nmf_weight,
            'gen_weight': gen_weight,
            'nmf_k': nmf_k,
            'ridge_alpha': ridge_alpha,
            'gpu_id': gpu_id,
            'use_attention': use_attention,
            'early_stopping_patience': early_stopping_patience,
            'min_epochs': min_epochs,
            # 동적 마스킹 옵션 추가
            'dynamic_masking': dynamic_masking,
            'mask_rate_range': mask_rate_range
        }
    else:
        # 사용자 지정 설정이 있으면 기본값과 병합
        default_config = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'n_critic': n_critic,
            'clip_value': clip_value,
            'mse_weight': mse_weight,
            'nmf_weight': nmf_weight,
            'gen_weight': gen_weight,
            'nmf_k': nmf_k,
            'ridge_alpha': ridge_alpha,
            'gpu_id': gpu_id,
            'use_attention': use_attention,
            'early_stopping_patience': early_stopping_patience,
            'min_epochs': min_epochs
        }
        # 사용자 설정으로 덮어쓰기
        default_config.update(model_config)
        # 동적 마스킹 옵션 추가 (사용자 설정에 없으면 기본값 사용)
        if 'dynamic_masking' not in model_config:
            default_config['dynamic_masking'] = dynamic_masking
        if 'mask_rate_range' not in model_config:
            default_config['mask_rate_range'] = mask_rate_range
        # src_combo 옵션 추가 (사용자 설정에 없으면 기본값 사용)
        if 'src_combo' not in model_config:
            default_config['src_combo'] = src_combo
        if 'tgt_modality' not in model_config:
            default_config['tgt_modality'] = tgt_modality
        model_config = default_config
    
    print("🔧 모델 설정:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # 데이터셋별 경로 자동 생성 (실제 파일명에 맞게 수정)
    data_variants = {
        "original": {
            "rna": f"{base_data_dir}/BRCA_PAM50.rna.original.tsv",
            "methy": f"{base_data_dir}/BRCA_PAM50.methy.original.tsv",
            "protein": f"{base_data_dir}/BRCA_PAM50.protein.original.tsv"
        },
        "missing_10p": {
            "rna": f"{base_data_dir}/BRCA_PAM50.missing_10p.rna.noisy.tsv",
            "methy": f"{base_data_dir}/BRCA_PAM50.missing_10p.methy.noisy.tsv",
            "protein": f"{base_data_dir}/BRCA_PAM50.missing_10p.protein.noisy.tsv"
        },
        "missing_30p": {
            "rna": f"{base_data_dir}/BRCA_PAM50.missing_30p.rna.noisy.tsv",
            "methy": f"{base_data_dir}/BRCA_PAM50.missing_30p.methy.noisy.tsv",
            "protein": f"{base_data_dir}/BRCA_PAM50.missing_30p.protein.noisy.tsv"
        },
        "missing_50p": {
            "rna": f"{base_data_dir}/BRCA_PAM50.missing_50p.rna.noisy.tsv",
            "methy": f"{base_data_dir}/BRCA_PAM50.missing_50p.methy.noisy.tsv",
            "protein": f"{base_data_dir}/BRCA_PAM50.missing_50p.protein.noisy.tsv"
        }
    }
    
    results = {}
    
    for variant_name, data_paths in data_variants.items():
        print(f"\n🚀 {variant_name.upper()} 데이터셋으로 모델 학습 시작")
        print("=" * 60)
        
        # 데이터 파일 존재 확인
        missing_files = []
        for omics, file_path in data_paths.items():
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ {variant_name} 데이터셋에 필요한 파일이 없습니다:")
            for file_path in missing_files:
                print(f"  - {file_path}")
            results[variant_name] = {"error": "필요한 데이터 파일이 없습니다"}
            continue
        
        # 결과 디렉토리 생성
        results_dir = f"{output_base_dir}/{variant_name}"
        os.makedirs(results_dir, exist_ok=True)
        
        # 데이터셋별 설정 생성
        variant_config = model_config.copy()
        variant_config.update({
            'data_dir': base_data_dir,
            'dataset_type': variant_name,
            'save_dir': results_dir,
            # src_combo와 tgt_modality 추가
            'src_combo': model_config.get('src_combo'),
            'tgt_modality': model_config.get('tgt_modality', 'protein')
        })
        
        try:
            # 학습기 생성 및 학습 시작
            trainer = MultiOmicsTrainer(variant_config)
            
            # 데이터 로드
            if not trainer.load_data():
                print(f"❌ {variant_name} 데이터 로드 실패")
                results[variant_name] = {"error": "데이터 로드 실패"}
                continue
            
            # 모델 학습
            trainer.train()
            
            # 결과 저장
            results[variant_name] = {
                "model_path": f"{results_dir}/model.pkl",
                "performance_path": f"{results_dir}/performance.tsv",
                "predictions_path": f"{results_dir}/predictions.tsv",
                "status": "success"
            }
            
            print(f"✅ {variant_name} 모델 학습 완료")
            print(f"📁 결과 저장 위치: {results_dir}")
            
        except Exception as e:
            print(f"❌ {variant_name} 모델 학습 실패: {e}")
            import traceback
            traceback.print_exc()
            results[variant_name] = {"error": str(e), "status": "failed"}
    
    return results


def compare_variants_performance(results: dict, output_file: str = "variant_comparison.tsv"):
    """
    모든 데이터셋별 성능 비교 및 요약
    
    Args:
        results: train_all_variants의 결과
        output_file: 비교 결과 저장 파일
    """
    print("\n📊 데이터셋별 성능 비교")
    print("=" * 60)
    
    comparison_data = []
    
    for variant_name, result in results.items():
        if result.get("status") == "success":
            # 성능 파일에서 결과 읽기
            performance_file = result["performance_path"]
            if os.path.exists(performance_file):
                try:
                    perf_df = pd.read_csv(performance_file, sep='\t')
                    # 마지막 에포크의 성능 추출
                    final_performance = perf_df.iloc[-1] if len(perf_df) > 0 else None
                    
                    if final_performance is not None:
                        comparison_data.append({
                            "variant": variant_name,
                            "final_epoch": final_performance.get("epoch", "N/A"),
                            "train_mse": final_performance.get("train_mse", "N/A"),
                            "val_mse": final_performance.get("val_mse", "N/A"),
                            "train_loss": final_performance.get("train_loss", "N/A"),
                            "val_loss": final_performance.get("val_loss", "N/A")
                        })
                        print(f"✅ {variant_name}: 검증 MSE = {final_performance.get('val_mse', 'N/A'):.6f}")
                    else:
                        print(f"⚠️ {variant_name}: 성능 데이터 없음")
                        
                except Exception as e:
                    print(f"❌ {variant_name}: 성능 파일 읽기 실패 - {e}")
            else:
                print(f"⚠️ {variant_name}: 성능 파일 없음")
        else:
            print(f"❌ {variant_name}: 학습 실패 - {result.get('error', '알 수 없는 오류')}")
    
    # 비교 결과를 데이터프레임으로 변환 및 저장
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_file, sep='\t', index=False)
        print(f"\n📊 성능 비교 결과 저장: {output_file}")
        print(comparison_df.to_string(index=False))
    else:
        print("❌ 비교할 성능 데이터가 없습니다.")


def main_auto_training():
    """자동 학습 모드"""
    print("🤖 자동 학습 모드 시작")
    
    # 기본 설정
    base_data_dir = "./processed_datasets"
    output_base_dir = "./results"
    
    # 모든 데이터셋 학습
    results = train_all_variants(
        base_data_dir=base_data_dir,
        model_config=None,
        output_base_dir=output_base_dir
    )
    
    # 성능 비교
    comparison_file = f"{output_base_dir}/variant_comparison.tsv"
    compare_variants_performance(results, comparison_file)
    
    print(f"\n🎉 모든 데이터셋 학습 완료!")
    print(f"📊 성능 비교 결과: {comparison_file}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='멀티오믹스 NMF+TGAN 학습')
    
    # 데이터 설정
    parser.add_argument('--data_dir', type=str, default='./processed_datasets',
                       help='전처리된 데이터셋 디렉토리')
    parser.add_argument('--dataset_type', type=str, default='original',
                       choices=['original', 'missing_10p', 'missing_30p', 'missing_50p'],
                       help='사용할 데이터셋 타입 (평탄화 파일 구조)')
    parser.add_argument('--src_combo', type=str, default=None,
                       help='소스 모달리티 조합(예: "rna+methyl", "protein+methyl"). 지정 시 concat 로더 사용')
    parser.add_argument('--tgt_modality', type=str, default='protein',
                       help='타깃 모달리티 (rna, methyl, protein)')
    
    # 모델 설정
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='학습률')
    parser.add_argument('--num_epochs', type=int, default=100, help='에포크 수')
    parser.add_argument('--n_critic', type=int, default=5, help='Critic 학습 횟수')
    parser.add_argument('--clip_value', type=float, default=0.01, help='WGAN 클리핑 값')
    
    # 손실 가중치
    parser.add_argument('--mse_weight', type=float, default=0.1, help='MSE 손실 가중치')
    parser.add_argument('--nmf_weight', type=float, default=0.1, help='NMF 손실 가중치')
    parser.add_argument('--gen_weight', type=float, default=1.0, help='Generator 손실 가중치')
    
    # NMF 설정
    parser.add_argument('--nmf_k', type=int, default=20, help='NMF 클러스터 수')
    parser.add_argument('--ridge_alpha', type=float, default=1e-3, help='Ridge 정규화 계수')
    
    # 기타
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='모델 저장 디렉토리')
    parser.add_argument('--use_attention', action='store_true', help='Cross-Attention 사용')
    
    # 동적 마스킹 옵션
    parser.add_argument('--dynamic_masking', action='store_true', help='배치마다 pseudo-missing 샘플링')
    parser.add_argument('--mask_rate_min', type=float, default=0.05, help='동적 마스킹 최소 비율')
    parser.add_argument('--mask_rate_max', type=float, default=0.50, help='동적 마스킹 최대 비율')
    
    # === Tri joint 학습 플래그 (추가) ===
    parser.add_argument('--tri_joint', action='store_true',
                       help='RNA/Protein/Methylation 3타깃을 한 번의 러닝으로 동시 학습')
    
    # 검증용 고정 마스크
    parser.add_argument('--val_mask_rate', type=float, default=0.30, help='검증 RMSE용 pseudo-missing 비율')
    parser.add_argument('--val_mask_seed', type=int, default=123, help='검증 RMSE용 시드')
    
    # 🚀 Early Stopping 설정
    parser.add_argument('--early_stopping_patience', type=int, default=10, 
                       help='Early Stopping 인내심 (에포크 수)')
    parser.add_argument('--min_epochs', type=int, default=5, 
                       help='최소 학습 에포크 수')
    
    # 🌱 시드 고정
    parser.add_argument('--seed', type=int, default=None, help='재현성을 위한 시드 고정')
    
    args = parser.parse_args()
    
    # 🌱 시드 고정 적용
    if args.seed is not None:
        set_seed(args.seed)
        print(f"🌱 시드 고정: {args.seed}")
    
    # 데이터셋 타입 검증
    data_dir = args.data_dir
    dataset_type = args.dataset_type
    
    # 사용 가능한 데이터셋 확인 (플랫 파일 구조 지원)
    available_datasets = []
    if os.path.exists(data_dir):
        # 평탄화 파일 구조 확인
        for dt in ['original', 'missing_10p', 'missing_30p', 'missing_50p']:
            # 각 데이터셋 타입에 대해 필요한 파일들이 존재하는지 확인
            required_files = [
                f"BRCA_PAM50.rna.{dt}.tsv",
                f"BRCA_PAM50.methy.{dt}.tsv",
                f"BRCA_PAM50.protein.{dt}.tsv"
            ]
            
            files_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)
            if files_exist:
                available_datasets.append(dt)
    
    print(f"사용 가능한 데이터셋: {available_datasets}")
    
    # 요청된 데이터셋이 사용 가능한지 확인
    if dataset_type not in available_datasets:
        if available_datasets:
            print(f"⚠️ 요청된 데이터셋 '{dataset_type}'이 없습니다.")
            print(f"사용 가능한 데이터셋 중 첫 번째를 사용합니다: {available_datasets[0]}")
            args.dataset_type = available_datasets[0]
        else:
            print("❌ 사용 가능한 데이터셋이 없습니다.")
            print(f"데이터 디렉토리 확인: {data_dir}")
            print("필요한 파일들(예):")
            for dt in ['original', 'missing_10p', 'missing_30p', 'missing_50p']:
                print(f"  - BRCA_PAM50.rna.{dt}.tsv")
                print(f"  - BRCA_PAM50.methy.{dt}.tsv")
                print(f"  - BRCA_PAM50.protein.{dt}.tsv")
            return
    
    # 설정 딕셔너리 생성
    config = vars(args)
    
    # src_combo 설정 출력
    if args.src_combo:
        print(f"🔍 src_combo: {args.src_combo}")
        print(f"🔍 tgt_modality: {args.tgt_modality}")
    
    # Tri-joint 설정 출력
    if args.tri_joint:
        print(f"🚀 Tri-joint 모드: RNA/Protein/Methylation 3타깃 동시 학습")
    
    # 동적 마스킹 설정 출력
    if args.dynamic_masking:
        print(f"동적 마스킹: U[{args.mask_rate_min:.2f}, {args.mask_rate_max:.2f}]")
    
    # Tri-joint 모드인 경우 별도 처리
    if args.tri_joint:
        # Tri-joint 모드: 3개 오믹스 동시 학습
        print("\n📊 Tri-joint 데이터 로딩 중...")
        try:
            train_loader, valid_loader, _, data_info = get_triple_dataloaders(
                data_dir=args.data_dir, batch_size=args.batch_size, split=(0.8,0.2)
            )
            print(f"✅ Tri-joint 데이터 로딩 성공!")
            print(f"RNA dim={data_info['src_dim_rna']}, Protein dim={data_info['src_dim_prot']}, Methyl dim={data_info['src_dim_methy']}")
            print(f"훈련 샘플: {data_info['n_train']} / 검증 샘플: {data_info['n_valid']}")
        except Exception as e:
            print(f"❌ Tri-joint 데이터 로딩 실패: {e}")
            return
        
        # Tri-joint 모델 생성
        print("\n🤖 Tri-joint 모델 생성 중...")
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        dim_r, dim_p, dim_m = data_info["src_dim_rna"], data_info["src_dim_prot"], data_info["src_dim_methy"]
        
        # Generators
        Gp = Generator(input_size=dim_r+dim_m, output_size=dim_p,
                       use_attn=args.use_attention, n_heads=4, d_head=64,
                       target_type="protein", src_size=dim_r+dim_m)
        Gr = Generator(input_size=dim_p+dim_m, output_size=dim_r,
                       use_attn=args.use_attention, n_heads=4, d_head=64,
                       target_type="rna", src_size=dim_p+dim_m)
        Gm = Generator(input_size=dim_r+dim_p, output_size=dim_m,
                       use_attn=args.use_attention, n_heads=4, d_head=64,
                       target_type="methyl", src_size=dim_r+dim_p)
        
        # Critics
        Dp = Critic(input_size=dim_p, hidden_dim=512)
        Dr = Critic(input_size=dim_r, hidden_dim=512)
        Dm = Critic(input_size=dim_m, hidden_dim=512)
        
        print("✅ Tri Generators/Critics 생성 완료")
        
        # Tri-joint 트레이너 생성
        trainer_tri = WGAN_GP_Trainer_Tri(
            Gp, Gr, Gm, Dp, Dr, Dm, device,
            lr_g=args.learning_rate, lr_d=args.learning_rate, 
            lambda_gp=10.0, n_critic=args.n_critic,
            dynamic_masking=args.dynamic_masking,
            mask_rate_range=(args.mask_rate_min, args.mask_rate_max)
        )
        
        # Tri-joint 학습 실행
        print(f"\n🚀 Tri-joint 학습 시작 (에포크: {args.num_epochs})")
        os.makedirs(args.save_dir, exist_ok=True)
        
        print(f"\n🚀 Tri-joint 학습 시작!")
        print(f"📊 총 {args.num_epochs} 에포크 | Early Stopping: {args.early_stopping_patience} 에포크")
        print("=" * 80)
        
        best_metric = float('inf')  # ↓ 평균 RMSE 기준으로 갱신
        patience_counter = 0
        
        for epoch in range(args.num_epochs):
            log = trainer_tri.train_epoch(train_loader)
            
            # 검증 RMSE(고정 마스크) 계산
            rmse = trainer_tri.validate_rmse(valid_loader, seed=args.val_mask_seed, mask_rate=args.val_mask_rate)
            avg_rmse = rmse["avg"]
            improved = avg_rmse < best_metric
            if improved:
                best_metric = avg_rmse
                patience_counter = 0
                status = "🏆"
                # best 저장
                ck_best = {
                    "Gp": Gp.state_dict(), "Gr": Gr.state_dict(), "Gm": Gm.state_dict(),
                    "Dp": Dp.state_dict(), "Dr": Dr.state_dict(), "Dm": Dm.state_dict(),
                    "epoch": epoch, "val_rmse": rmse
                }
                torch.save(ck_best, os.path.join(args.save_dir, "tri_best.ckpt"))
            else:
                patience_counter += 1
                status = "📈" if patience_counter < args.early_stopping_patience else "⚠️"
            
            # 에포크 로그
            print(f"Epoch {epoch+1:3d}/{args.num_epochs} | {status} G: {log['G']:7.3f} | D: {log['D']:8.3f} "
                  f"| Val RMSE avg={avg_rmse:.6f} (P={rmse['protein']:.6f}, R={rmse['rna']:.6f}, M={rmse['methyl']:.6f}) "
                  f"| Best={best_metric:.6f}")
            
            # 주기적 저장
            if (epoch + 1) % 10 == 0 or (epoch+1)==args.num_epochs:
                ck = {
                    "Gp": Gp.state_dict(), "Gr": Gr.state_dict(), "Gm": Gm.state_dict(),
                    "Dp": Dp.state_dict(), "Dr": Dr.state_dict(), "Dm": Dm.state_dict(),
                    "epoch": epoch,
                }
                torch.save(ck, os.path.join(args.save_dir, f"tri_epoch_{epoch+1}.ckpt"))
                print(f"💾 체크포인트 저장: Epoch {epoch+1}")
            
            # Early Stopping 체크
            if epoch >= args.min_epochs and patience_counter >= args.early_stopping_patience:
                print(f"\n🛑 Early Stopping! {args.early_stopping_patience} 에포크 동안 개선 없음")
                break
        
        # 마지막 tri 가중치 저장
        ck = {
            "Gp": Gp.state_dict(), "Gr": Gr.state_dict(), "Gm": Gm.state_dict(),
            "Dp": Dp.state_dict(), "Dr": Dr.state_dict(), "Dm": Dm.state_dict(),
            "epoch": args.num_epochs - 1,
        }
        torch.save(ck, os.path.join(args.save_dir, "tri_final.ckpt"))
        print(f"💾 Tri-joint 최종 모델 저장: tri_final.ckpt (best=tri_best.ckpt, 기준: 검증 평균 RMSE)")
        print("🎉 Tri-joint 학습 완료!")
        return
    
    # 기존 단일 타깃 학습 경로
    print("\n📊 기존 단일 타깃 학습 모드")
    trainer = MultiOmicsTrainer(config)
    trainer.train()


# ========================================
# 데이터셋별 자동 학습 함수들
# ========================================

def main_auto_training():
    """
    자동 학습을 위한 메인 함수 (CLI에서 사용)
    """
    parser = argparse.ArgumentParser(description='모든 데이터셋 변형에 대한 자동 학습')
    
    # 데이터 설정
    parser.add_argument('--data_dir', type=str, 
                       default='/home/dyan/nmf/mochi_code/process_data',
                       help='전처리된 데이터셋 디렉토리')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='결과 저장 디렉토리')
    
    # 모델 설정 (기본값)
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='학습률')
    parser.add_argument('--num_epochs', type=int, default=100, help='에포크 수')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    
    # 동적 마스킹 옵션
    parser.add_argument('--dynamic_masking', action='store_true', help='배치마다 pseudo-missing 샘플링')
    parser.add_argument('--mask_rate_min', type=float, default=0.05, help='동적 마스킹 최소 비율')
    parser.add_argument('--mask_rate_max', type=float, default=0.50, help='동적 마스킹 최대 비율')
    
    # src_combo 옵션
    parser.add_argument('--src_combo', type=str, default=None, help='소스 모달리티 조합')
    parser.add_argument('--tgt_modality', type=str, default='protein', help='타깃 모달리티')
    
    args = parser.parse_args()
    
    # 모델 설정
    model_config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'gpu_id': args.gpu_id,
        'n_critic': 5,
        'clip_value': 0.01,
        'mse_weight': 0.1,
        'nmf_weight': 0.1,
        'gen_weight': 1.0,
        'nmf_k': 20,
        'ridge_alpha': 1e-3,
        'use_attention': False,
        'early_stopping_patience': 10,
        'min_epochs': 5,
        # 동적 마스킹 옵션
        'dynamic_masking': args.dynamic_masking,
        'mask_rate_range': (args.mask_rate_min, args.mask_rate_max),
        # src_combo 옵션
        'src_combo': args.src_combo,
        'tgt_modality': args.tgt_modality
    }
    
    print("🚀 모든 데이터셋 변형에 대한 자동 학습 시작")
    print(f"📁 데이터 디렉토리: {args.data_dir}")
    print(f"📁 결과 디렉토리: {args.output_dir}")
    
    # 모든 데이터셋 학습
    results = train_all_variants(
        base_data_dir=args.data_dir,
        model_config=model_config,
        output_base_dir=args.output_dir
    )
    
    # 성능 비교
    comparison_file = f"{args.output_dir}/variant_comparison.tsv"
    compare_variants_performance(results, comparison_file)
    
    print(f"\n🎉 모든 데이터셋 학습 완료!")
    print(f"📊 성능 비교 결과: {comparison_file}")


def quick_train_all_variants(
    base_data_dir: str = "/home/dyan/nmf/mochi_code/process_data",
    output_base_dir: str = "results",
    # 동적 마스킹 옵션 추가
    dynamic_masking: bool = False,
    mask_rate_range: tuple = (0.05, 0.5),
    # src_combo 옵션 추가
    src_combo: str = None,
    tgt_modality: str = 'protein',
    **kwargs
):
    """
    주피터에서 간단하게 사용할 수 있는 모든 데이터셋 학습 함수
    
    Args:
        base_data_dir: 전처리된 데이터가 있는 기본 디렉토리
        output_base_dir: 결과 저장 기본 디렉토리
        **kwargs: 모델 파라미터들 (num_epochs, batch_size, learning_rate 등)
    
    Returns:
        dict: 각 데이터셋별 학습 결과
    """
    
    print("🚀 모든 데이터셋 변형에 대한 빠른 학습 시작")
    print(f"📁 데이터 디렉토리: {base_data_dir}")
    print(f"📁 결과 디렉토리: {output_base_dir}")
    
    # 기본값 설정
    default_params = {
        'num_epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'early_stopping_patience': 8,
        'min_epochs': 5,
        'gpu_id': 0,
        # 동적 마스킹 옵션
        'dynamic_masking': dynamic_masking,
        'mask_rate_range': mask_rate_range,
        # src_combo 옵션
        'src_combo': src_combo,
        'tgt_modality': tgt_modality
    }
    
    # 사용자 지정 파라미터로 덮어쓰기
    default_params.update(kwargs)
    
    print("🔧 학습 파라미터:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")
    
    # 학습 실행
    results = train_all_variants(
        base_data_dir=base_data_dir,
        model_config=None,  # 함수 파라미터 사용
        output_base_dir=output_base_dir,
        dynamic_masking=dynamic_masking,
        mask_rate_range=mask_rate_range,
        src_combo=src_combo,
        tgt_modality=tgt_modality,
        **default_params
    )
    
    # 성능 비교
    comparison_file = f"{output_base_dir}/variant_comparison.tsv"
    compare_variants_performance(results, comparison_file)
    
    print(f"\n🎉 모든 데이터셋 학습 완료!")
    print(f"📊 성능 비교 결과: {comparison_file}")
    
    return results


if __name__ == "__main__":
    # 기존 main() 함수는 유지하고, 자동 학습을 위한 새로운 진입점 추가
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        # 자동 학습 모드
        sys.argv.pop(1)  # --auto 제거
        main_auto_training()
    else:
        # 기존 학습 모드
        main()
