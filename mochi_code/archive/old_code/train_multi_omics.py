#!/usr/bin/env python3
"""
멀티오믹스 NMF+TGAN 학습 스크립트
- models.py의 Generator/Critic과 연결
- dataloader.py의 MultiOmicsDataset 사용
- utils.py의 평가 함수 활용
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# 로컬 모듈 임포트
from models import Generator, Critic
from dataloader import get_multi_omics_dataloaders
from utils import evaluate_model_performance, calculate_imputation_metrics

class WGAN_GP_Trainer:
    """
    WGAN-GP 기반 멀티오믹스 학습기
    """
    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        device: torch.device,
        lr_g: float = 0.0001,
        lr_d: float = 0.0001,
        lambda_gp: float = 10.0,
        n_critic: int = 5
    ):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        
        # 옵티마이저
        self.optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.optimizer_d = optim.Adam(critic.parameters(), lr=lr_d, betas=(0.5, 0.9))
        
        # 손실 함수
        self.criterion = nn.MSELoss()
        
        # 학습 히스토리
        self.history = {
            'g_loss': [], 'd_loss': [], 'gp_loss': [],
            'train_rmse': [], 'valid_rmse': [],
            'train_mae': [], 'valid_mae': [],
            'train_pearson': [], 'valid_pearson': []
        }
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Gradient Penalty 계산 (WGAN-GP)"""
        alpha = torch.rand(real_samples.size(0), 1, device=self.device)
        alpha = alpha.expand(real_samples.size())
        
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        d_interpolates = self.critic(interpolates)
        
        fake = torch.ones(d_interpolates.size(), device=self.device, requires_grad=False)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def train_step(self, batch):
        """한 스텝 학습"""
        x_src = batch['x_src'].to(self.device)
        y_true = batch['y_true'].to(self.device)
        mask = batch['mask'].to(self.device)
        
        batch_size = x_src.size(0)
        
        # ===== Critic 학습 =====
        for _ in range(self.n_critic):
            self.optimizer_d.zero_grad()
            
            # 실제 데이터
            real_validity = self.critic(y_true)
            
            # 가짜 데이터 생성
            fake_data = self.generator(x_src, src=None)
            fake_validity = self.critic(fake_data.detach())
            
            # Gradient Penalty
            gradient_penalty = self.compute_gradient_penalty(y_true, fake_data.detach())
            
            # Critic 손실
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
            
            d_loss.backward()
            self.optimizer_d.step()
        
        # ===== Generator 학습 =====
        self.optimizer_g.zero_grad()
        
        # 가짜 데이터 재생성
        fake_data = self.generator(x_src, src=None)
        fake_validity = self.critic(fake_data)
        
        # Generator 손실 (WGAN + MSE)
        g_loss = -torch.mean(fake_validity)
        
        # 결측치 위치에서만 MSE 손실 추가
        if mask.sum() > 0:
            mse_loss = self.criterion(fake_data * mask, y_true * mask)
            g_loss = g_loss + 0.1 * mse_loss  # 가중치 조절 가능
        
        g_loss.backward()
        self.optimizer_g.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'gp_loss': gradient_penalty.item()
        }
    
    def train_epoch(self, train_loader, valid_loader):
        """한 에포크 학습"""
        self.generator.train()
        self.critic.train()
        
        epoch_losses = {'g_loss': [], 'd_loss': [], 'gp_loss': []}
        
        # 훈련
        for batch in tqdm(train_loader, desc="Training"):
            losses = self.train_step(batch)
            for k, v in losses.items():
                epoch_losses[k].append(v)
        
        # 검증
        valid_metrics = self.evaluate(valid_loader)
        
        # 히스토리 업데이트
        for k in epoch_losses:
            self.history[k].append(np.mean(epoch_losses[k]))
        
        self.history['valid_rmse'].append(valid_metrics['rmse'])
        self.history['valid_mae'].append(valid_metrics['mae'])
        self.history['valid_pearson'].append(valid_metrics['pearson'])
        
        return {
            'g_loss': np.mean(epoch_losses['g_loss']),
            'd_loss': np.mean(epoch_losses['d_loss']),
            'gp_loss': np.mean(epoch_losses['gp_loss']),
            'valid_rmse': valid_metrics['rmse'],
            'valid_mae': valid_metrics['mae'],
            'valid_pearson': valid_metrics['pearson']
        }
    
    def evaluate(self, data_loader):
        """모델 평가"""
        metrics, _, _, _ = evaluate_model_performance(
            self.generator, data_loader, self.device
        )
        return metrics
    
    def save_checkpoint(self, path, epoch, metrics):
        """체크포인트 저장"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'history': self.history,
            'metrics': metrics
        }, path)
    
    def load_checkpoint(self, path):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['metrics']

def main():
    parser = argparse.ArgumentParser(description='멀티오믹스 NMF+TGAN 학습')
    
    # 데이터 설정
    parser.add_argument('--data_dir', type=str, default='./processed_datasets',
                       help='전처리된 데이터 디렉토리')
    parser.add_argument('--src_modality', type=str, default='rna',
                       help='소스 모달리티 (rna, methyl, protein)')
    parser.add_argument('--tgt_modality', type=str, default='protein',
                       help='타깃 모달리티 (rna, methyl, protein)')
    
    # 모델 설정
    parser.add_argument('--use_attention', action='store_true',
                       help='Cross-Attention 사용')
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
    
    # 기타 설정
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--device', type=str, default='auto',
                       help='디바이스 (auto, cuda, cpu)')
    parser.add_argument('--resume', type=str, default=None,
                       help='체크포인트 경로 (재시작용)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 멀티오믹스 NMF+TGAN 학습 시작")
    print(f"디바이스: {device}")
    print(f"소스: {args.src_modality} → 타깃: {args.tgt_modality}")
    print(f"Cross-Attention: {'사용' if args.use_attention else '미사용'}")
    
    # 데이터로더 생성
    print("\n📊 데이터 로딩 중...")
    try:
        train_loader, valid_loader, _, data_info = get_multi_omics_dataloaders(
            data_dir=args.data_dir,
            src_modality=args.src_modality,
            tgt_modality=args.tgt_modality,
            batch_size=args.batch_size
        )
        print(f"✅ 데이터 로딩 성공!")
        print(f"소스 차원: {data_info['src_dim']}")
        print(f"타깃 차원: {data_info['tgt_dim']}")
        print(f"훈련 샘플: {data_info['n_train']}")
        print(f"검증 샘플: {data_info['n_valid']}")
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return
    
    # 모델 생성
    print("\n🤖 모델 생성 중...")
    
    # target_type 자동 설정
    target_type = args.tgt_modality
    
    generator = Generator(
        input_size=data_info['src_dim'],
        output_size=data_info['tgt_dim'],
        use_attn=args.use_attention,
        n_heads=args.n_heads,
        d_head=args.d_head,
        n_src_tokens=args.n_src_tokens,
        target_type=target_type,
        src_size=data_info['src_dim']
    )
    
    critic = Critic(
        input_size=data_info['tgt_dim'],
        hidden_dim=512
    )
    
    print(f"✅ 모델 생성 완료!")
    print(f"Generator 파라미터: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Critic 파라미터: {sum(p.numel() for p in critic.parameters()):,}")
    
    # 학습기 생성
    trainer = WGAN_GP_Trainer(
        generator=generator,
        critic=critic,
        device=device,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_gp=args.lambda_gp,
        n_critic=args.n_critic
    )
    
    # 체크포인트 재시작
    start_epoch = 0
    if args.resume:
        print(f"🔄 체크포인트에서 재시작: {args.resume}")
        start_epoch, _ = trainer.load_checkpoint(args.resume)
        start_epoch += 1
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 학습 루프
    print(f"\n🎯 학습 시작 (에포크 {start_epoch} ~ {args.epochs})")
    best_rmse = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # 학습
        metrics = trainer.train_epoch(train_loader, valid_loader)
        
        # 결과 출력
        print(f"Generator Loss: {metrics['g_loss']:.4f}")
        print(f"Critic Loss: {metrics['d_loss']:.4f}")
        print(f"Gradient Penalty: {metrics['gp_loss']:.4f}")
        print(f"Validation RMSE: {metrics['valid_rmse']:.4f}")
        print(f"Validation MAE: {metrics['valid_mae']:.4f}")
        print(f"Validation Pearson: {metrics['valid_pearson']:.4f}")
        
        # 최고 성능 체크포인트 저장
        if metrics['valid_rmse'] < best_rmse:
            best_rmse = metrics['valid_rmse']
            best_path = os.path.join(args.output_dir, 'model_best.ckpt')
            trainer.save_checkpoint(best_path, epoch, metrics)
            print(f"🏆 새로운 최고 성능! RMSE: {best_rmse:.4f}")
        
        # 주기적 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'model_epoch_{epoch+1}.ckpt')
            trainer.save_checkpoint(checkpoint_path, epoch, metrics)
    
    # 최종 모델 저장
    final_path = os.path.join(args.output_dir, 'model_final.ckpt')
    trainer.save_checkpoint(final_path, args.epochs - 1, metrics)
    
    # 학습 히스토리 저장
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)
    
    print(f"\n🎉 학습 완료!")
    print(f"결과 저장 위치: {args.output_dir}")
    print(f"최고 성능: RMSE {best_rmse:.4f}")

if __name__ == '__main__':
    main()
