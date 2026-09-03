# MOCHI

TCGA 멀티오믹스 **칸 결측**과 **블록 결측**을 한 모형에서 채우는 연구 코드입니다. OmicsNMF(NMF + GAN)를 세 오믹스 상호보정으로 확장합니다.

보고 모형은 `mochi_code/train_nmf_tf.py` (오토인코더 은닉 + 고정 NMF 토큰 + 저랭크 잔차 + 조건부 WGAN)입니다.

## 폴더

```
mochi_code/     현재 모델과 실험 (여기만 수정)
baselines/      공식 OmicsNMF, OmiTrans, MIMIR (비교군, 수정하지 않음, gitignore)
archive/        예전 파이프라인, 노트북, 툴킷
```

학습은 **NVIDIA GPU가 있는 리눅스/윈도우**에서만 합니다. MacBook(에어/프로, Apple Silicon)에는 CUDA가 없습니다. 맥에서 `cu124` 인덱스로 torch를 받으면 패키지를 찾지 못합니다.

맥에서는 데이터만 받아도 됩니다. 학습은 NVIDIA 머신에서 이어서 합니다.

```bash
git clone https://github.com/llovable/nmf.git
cd nmf
git checkout cursor/align-manuscript-with-lowrank-path-db76

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# 맥: torch 설치하지 말 것. fetch만 가능.
# NVIDIA 머신: https://pytorch.org 에서 CUDA 휠. 예
#   pip install torch --index-url https://download.pytorch.org/whl/cu124

cd mochi_code
bash scripts/fetch_brca.sh
# 아래는 nvidia-smi 가 GPU를 찍을 때만
bash scripts/run_next_experiments.sh
```

그림 1은 `mochi_code/paper/figures/fig1_pathway_knockout.pdf`입니다.

## 커밋

- 모델·논문: `mochi_code/`만 스테이징 (결과·전처리 행렬은 gitignore)
- 공식 베이스라인은 `baselines/`에 그대로 둠
