import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os

class MultiOmicsDataset(Dataset):
    """
    멀티오믹스 데이터셋 (RNA, Methylation, Protein)
    - src: 소스 오믹스 (예: RNA)
    - tgt: 타깃 오믹스 (예: Protein)
    - mask: 타깃 마스크 (1=missing, 0=observed)
    """
    def __init__(
        self, 
        data_dict: Dict,
        samples: List[str],
        src_modality: str = "rna",
        tgt_modality: str = "protein"
    ):
        self.samples = samples
        self.src_modality = src_modality
        self.tgt_modality = tgt_modality
        
        # 소스 데이터 (예: RNA)
        self.X_src = data_dict[f'train_{src_modality}']
        
        # 타깃 데이터 (예: Protein)
        self.Y_tgt = data_dict[f'train_{tgt_modality}']
        
        # 마스크 (1=missing, 0=observed)
        self.M_tgt = data_dict[f'train_{tgt_modality}_mask']
        
        # 특징 차원
        self.src_dim = self.X_src.shape[1]
        self.tgt_dim = self.Y_tgt.shape[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 소스 데이터 (예: RNA)
        x_src = self.X_src.loc[sample].values.astype(np.float32)
        
        # 타깃 데이터 (예: Protein)
        y_tgt = self.Y_tgt.loc[sample].values.astype(np.float32)
        
        # 마스크 (1=missing, 0=observed)
        mask = self.M_tgt.loc[sample].values.astype(np.float32)
        
        # 결측치 처리: 결측 위치는 0으로, 관측 위치는 원본 값
        y_obs = np.where(mask == 0, y_tgt, 0.0)
        
        return {
            'sample': sample,
            'x_src': x_src,      # [src_dim]
            'y_tgt': y_obs,      # [tgt_dim] - 관측된 값만
            'mask': mask,         # [tgt_dim] - 1=missing, 0=observed
            'y_true': y_tgt      # [tgt_dim] - 원본 값 (평가용)
        }

def get_multi_omics_dataloaders(
    data_dir: str,
    src_modality: str = "rna",
    tgt_modality: str = "protein",
    batch_size: int = 32,
    train_ratio: float = 0.8,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    멀티오믹스 데이터로더 생성
    
    Args:
        data_dir: 전처리된 데이터 디렉토리
        src_modality: 소스 모달리티 ("rna", "methyl", "protein")
        tgt_modality: 타깃 모달리티 ("rna", "methyl", "protein")
        batch_size: 배치 크기
        train_ratio: 훈련/검증 분할 비율
        random_state: 랜덤 시드
    
    Returns:
        train_loader, valid_loader, test_loader, data_info
    """
    
    # 데이터 파일 경로
    src_file = os.path.join(data_dir, f"{src_modality}.tsv")
    tgt_file = os.path.join(data_dir, f"{tgt_modality}.tsv")
    mask_file = os.path.join(data_dir, f"{tgt_modality}_mask.tsv")
    
    # 파일 존재 확인
    for file_path in [src_file, tgt_file, mask_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
    
    # 데이터 로드
    X_src = pd.read_csv(src_file, sep='\t', index_col=0)      # 소스 (예: RNA)
    Y_tgt = pd.read_csv(tgt_file, sep='\t', index_col=0)      # 타깃 (예: Protein)
    M_tgt = pd.read_csv(mask_file, sep='\t', index_col=0)     # 마스크
    
    # 공통 샘플 찾기
    common_samples = X_src.index.intersection(Y_tgt.index).intersection(M_tgt.index)
    print(f"공통 샘플 수: {len(common_samples)}")
    
    # 데이터 정렬
    X_src = X_src.loc[common_samples]
    Y_tgt = Y_tgt.loc[common_samples]
    M_tgt = M_tgt.loc[common_samples]
    
    # 훈련/검증 분할
    train_samples, valid_samples = train_test_split(
        common_samples, 
        train_size=train_ratio, 
        random_state=random_state
    )
    
    # 데이터셋 생성
    data_dict = {
        f'train_{src_modality}': X_src,
        f'train_{tgt_modality}': Y_tgt,
        f'train_{tgt_modality}_mask': M_tgt
    }
    
    train_dataset = MultiOmicsDataset(
        data_dict, train_samples, src_modality, tgt_modality
    )
    valid_dataset = MultiOmicsDataset(
        data_dict, valid_samples, src_modality, tgt_modality
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_multi_omics
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_multi_omics
    )
    
    # 데이터 정보
    data_info = {
        'src_dim': X_src.shape[1],
        'tgt_dim': Y_tgt.shape[1],
        'src_modality': src_modality,
        'tgt_modality': tgt_modality,
        'n_samples': len(common_samples),
        'n_train': len(train_samples),
        'n_valid': len(valid_samples)
    }
    
    return train_loader, valid_loader, None, data_info

def collate_multi_omics(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    배치 데이터를 텐서로 변환
    """
    samples = [item['sample'] for item in batch]
    x_src = torch.stack([torch.tensor(item['x_src']) for item in batch])
    y_tgt = torch.stack([torch.tensor(item['y_tgt']) for item in batch])
    mask = torch.stack([torch.tensor(item['mask']) for item in batch])
    y_true = torch.stack([torch.tensor(item['y_true']) for item in batch])
    
    return {
        'samples': samples,
        'x_src': x_src,      # [B, src_dim]
        'y_tgt': y_tgt,      # [B, tgt_dim]
        'mask': mask,         # [B, tgt_dim]
        'y_true': y_true     # [B, tgt_dim]
    }

def get_target_type(modality: str) -> str:
    """
    모달리티에 따른 target_type 반환
    """
    if modality == "methyl":
        return "methyl"
    elif modality == "rna":
        return "rna"
    elif modality == "protein":
        return "protein"
    else:
        return "rna"  # 기본값

class ConcatPairDataset(Dataset):
    """
    src_combo에 있는 모달리티 테이블을 열 방향으로 concat하여 x_src를 만들고,
    tgt_modality 테이블을 y_true로 사용한다.
    mask는 y_true.isna().astype(float)로 생성(1=missing).
    """
    def __init__(self, tables: dict[str, pd.DataFrame], src_combo: list[str], tgt_modality: str):
        # 공통 index/column 정렬
        idx = None
        for m in src_combo + [tgt_modality]:
            idx = tables[m].index if idx is None else idx.intersection(tables[m].index)
        for m in src_combo + [tgt_modality]:
            tables[m] = tables[m].loc[idx]

        self.X_src = pd.concat([tables[m] for m in src_combo], axis=1)
        self.Y_tgt = tables[tgt_modality]
        self.mask = self.Y_tgt.isna().astype(np.float32)

        # NaN은 학습 전 단계에서만 존재 가능. DataLoader로 넘길 땐 임시 0 채움.
        self.Y_tgt_fill = self.Y_tgt.fillna(0.0).astype(np.float32)
        self.X_src = self.X_src.astype(np.float32)

    def __len__(self): 
        return self.X_src.shape[0]

    def __getitem__(self, i):
        return {
            "x_src": torch.from_numpy(self.X_src.iloc[i].to_numpy()),
            "y_true": torch.from_numpy(self.Y_tgt_fill.iloc[i].to_numpy()),
            "mask": torch.from_numpy(self.mask.iloc[i].to_numpy()),
        }

def _read_tsv(path):
    """TSV 파일 읽기"""
    return pd.read_csv(path, sep="\t", index_col=0)

def get_multi_omics_dataloaders_concat(
    data_dir,
    src_combo,          # 예: ["rna","methy"] 또는 ["rna","protein"]
    tgt_modality,             # "protein" | "rna" | "methyl"
    prefix="BRCA_PAM50",    # 파일 접두사
    split=(0.8, 0.2),
    batch_size=32,
    shuffle=True,
):
    """
    여러 소스 모달리티를 concat하여 데이터로더 생성
    
    Args:
        data_dir: 데이터 디렉토리
        src_combo: 소스 모달리티 리스트 (예: ["rna", "methy"])
        tgt_modality: 타깃 모달리티
        prefix: 파일 접두사
        split: 훈련/검증 분할 비율
        batch_size: 배치 크기
        shuffle: 훈련 데이터 셔플 여부
    
    Returns:
        train_loader, valid_loader, None, data_info
    """
    from pathlib import Path
    
    data_dir = Path(data_dir)
    needed = set(src_combo + [tgt_modality])
    paths = {}
    
    for m in needed:
        # 파일명 매칭 (methy vs methyl)
        if m == 'methy':
            fn = f"{prefix}.methy.original.tsv"
        elif m == 'methyl':
            fn = f"{prefix}.methy.original.tsv"  # 실제 파일명은 methy
        else:
            fn = f"{prefix}.{m}.original.tsv"
            
        p = data_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"파일 없음: {p}")
        paths[m] = p

    tables = {m: _read_tsv(p) for m, p in paths.items()}
    ds = ConcatPairDataset(tables, src_combo=src_combo, tgt_modality=tgt_modality)

    n = len(ds)
    n_train = int(n * split[0])
    n_valid = n - n_train
    
    gen = torch.Generator().manual_seed(42)
    train_ds, valid_ds = torch.utils.data.random_split(ds, [n_train, n_valid], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    src_dim = sum([tables[m].shape[1] for m in src_combo])
    tgt_dim = tables[tgt_modality].shape[1]
    
    data_info = {
        "src_combo": "+".join(src_combo),
        "tgt_modality": tgt_modality,
        "src_dim": src_dim,
        "tgt_dim": tgt_dim,
        "n_train": n_train,
        "n_valid": n_valid,
    }
    
    return train_loader, valid_loader, None, data_info

# === Tri-modal dataloader for joint training (추가) ==========================
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def _read_tsv(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)

class TriModalDataset(Dataset):
    """
    RNA / Protein / Methylation 세 테이블을 동시에 로드하여
    배치마다 x_rna, x_prot, x_methy와 실제 NA 마스크(m_rna/m_prot/m_methy)를 반환.
    - 마스크: 1=실제 NA, 0=관측
    - 타깃 NaN은 0으로 채워 텐서화, 손실은 마스크로 제어
    """
    def __init__(self, data_dir, prefix="BRCA_PAM50"):
        data_dir = Path(data_dir)
        
        # source/target 파일 자동 탐지 및 매핑
        def find_best_file(modality):
            # 가능한 파일 패턴들 (우선순위 순)
            patterns = [
                f"{prefix}.{modality}.original.tsv",  # 직접 original
                f"{prefix}.{modality}.original.source.tsv",  # source 파일
                f"{prefix}.{modality}.original.target.tsv",  # target 파일
                f"{prefix}.{modality}_to_*.original.source.tsv",  # _to_ 포함 source
                f"{prefix}.{modality}_to_*.original.target.tsv",  # _to_ 포함 target
                f"{prefix}.{modality}_to_*.original.tsv",  # _to_ 포함 일반
            ]
            
            for pattern in patterns:
                matches = list(data_dir.glob(pattern))
                if matches:
                    # missing_* 파일은 제외 (original만 사용)
                    valid_matches = [m for m in matches if "missing_" not in m.name]
                    if valid_matches:
                        return valid_matches[0]
            return None
        
        # 각 모달리티별로 최적 파일 찾기
        rna_file = find_best_file("rna")
        prot_file = find_best_file("protein") 
        methy_file = find_best_file("methy")
        
        if not all([rna_file, prot_file, methy_file]):
            raise FileNotFoundError(f"필요한 파일을 찾을 수 없습니다: rna={rna_file}, protein={prot_file}, methy={methy_file}")
        
        print(f"🔍 자동 파일 매핑:")
        print(f"  RNA: {rna_file.name}")
        print(f"  Protein: {prot_file.name}")
        print(f"  Methylation: {methy_file.name}")
        
        # 데이터 로드 및 전치 (행↔열 전치: 특징↔환자)
        print(f"🔄 데이터 전치 중...")
        self.rna   = _read_tsv(rna_file).T.astype(np.float32)    # 행↔열 전치
        self.prot  = _read_tsv(prot_file).T.astype(np.float32)   # 행↔열 전치
        self.methy = _read_tsv(methy_file).T.astype(np.float32)  # 행↔열 전치
        
        print(f"📊 전치 후 데이터 크기:")
        print(f"  RNA: {self.rna.shape} (환자 × 특징)")
        print(f"  Protein: {self.prot.shape} (환자 × 특징)")
        print(f"  Methylation: {self.methy.shape} (환자 × 특징)")
        
        # 샘플 수 확인 (환자 수)
        if self.rna.shape[0] == 0 or self.prot.shape[0] == 0 or self.methy.shape[0] == 0:
            raise ValueError(f"데이터가 비어있습니다: RNA={self.rna.shape[0]}, Protein={self.prot.shape[0]}, Methylation={self.methy.shape[0]}")
        
        # 공통 환자 샘플 정렬 (행 인덱스 = 환자 ID)
        idx = self.rna.index.intersection(self.prot.index).intersection(self.methy.index)
        self.rna, self.prot, self.methy = self.rna.loc[idx], self.prot.loc[idx], self.methy.loc[idx]
        # 실제 NA 마스크(1=NA)
        self.m_rna   = self.rna.isna().astype(np.float32)
        self.m_prot  = self.prot.isna().astype(np.float32)
        self.m_methy = self.methy.isna().astype(np.float32)
        # NaN 임시 채움(학습 시 마스크로 손실 제어)
        self.rna_f   = self.rna.fillna(0.0)
        self.prot_f  = self.prot.fillna(0.0)
        self.methy_f = self.methy.fillna(0.0)

    def __len__(self): 
        return self.rna.shape[0]

    def __getitem__(self, i):
        return {
            "x_rna":   torch.from_numpy(self.rna_f.iloc[i].to_numpy()),
            "x_prot":  torch.from_numpy(self.prot_f.iloc[i].to_numpy()),
            "x_methy": torch.from_numpy(self.methy_f.iloc[i].to_numpy()),
            "m_rna":   torch.from_numpy(self.m_rna.iloc[i].to_numpy()),
            "m_prot":  torch.from_numpy(self.m_prot.iloc[i].to_numpy()),
            "m_methy": torch.from_numpy(self.m_methy.iloc[i].to_numpy()),
        }

def get_triple_dataloaders(
    data_dir,
    batch_size=32,
    split=(0.8, 0.2),
    shuffle=True,
    prefix="BRCA_PAM50",
):
    """
    RNA/Protein/Methylation 3모달 동시 학습용 로더 반환
    """
    ds = TriModalDataset(data_dir, prefix=prefix)
    n = len(ds)
    n_tr = int(n * split[0])
    n_va = n - n_tr
    
    gen = torch.Generator().manual_seed(42)
    tr, va = torch.utils.data.random_split(ds, [n_tr, n_va], generator=gen)
    
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=shuffle, drop_last=False),
        DataLoader(va, batch_size=batch_size, shuffle=False, drop_last=False),
        None,
        {
            "src_dim_rna":   ds.rna.shape[1],
            "src_dim_prot":  ds.prot.shape[1],
            "src_dim_methy": ds.methy.shape[1],
            "n_train": n_tr,
            "n_valid": n_va,
        },
    )

if __name__ == '__main__':
    # 테스트
    try:
        train_loader, valid_loader, _, data_info = get_multi_omics_dataloaders(
            data_dir="./processed_datasets",
            src_modality="rna",
            tgt_modality="protein",
            batch_size=16
        )
        
        print("✅ 데이터로더 생성 성공!")
        print(f"소스 차원: {data_info['src_dim']}")
        print(f"타깃 차원: {data_info['tgt_dim']}")
        print(f"훈련 샘플: {data_info['n_train']}")
        print(f"검증 샘플: {data_info['n_valid']}")
        
        # 첫 번째 배치 확인
        for batch in train_loader:
            print(f"\n배치 크기: {batch['x_src'].shape}")
            print(f"소스 데이터: {batch['x_src'].shape}")
            print(f"타깃 데이터: {batch['y_tgt'].shape}")
            print(f"마스크: {batch['mask'].shape}")
            break
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("데이터 디렉토리와 파일들을 확인해주세요.")
