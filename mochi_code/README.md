# mochi_code

보고 모형은 NMF-Transformer(`train_nmf_tf.py`)입니다. 예전 NMF+TGAN·게이트 번역기는 `archive/`에 있습니다.

## 활성 파일

| 파일 | 역할 |
|---|---|
| `prepare_gate_data.py` | TCGA 종양 정렬, train/val/test |
| `models_nmf_tf.py` | 보고 모형: AE 은닉 + NMF 토큰 + 저랭크 잔차 |
| `train_nmf_tf.py` | **현재 학습** |
| `eval_biology.py` | 경로·환자구조·교차오믹스 보존 |
| `eval_amplitude.py` | 경로 진폭 수축 → 군집·차등발현 |
| `eval_mcar_mnar.py` | 칸 결측 MCAR/MNAR |
| `official_wrap.py` / `mimir_wrap.py` | 비교군 래퍼 |
| `models.py` | Generator / ConditionalCritic (공용) |
| `evaluate_imputation.py` | 가린 칸만 채점 |

## 데이터·결과

```
processed_data/gate_{brca,luad,kirc}/   전처리 스플릿 (gitignore)
results/current/                        학습·평가 (gitignore)
paper/                                  초고와 그림 1
archive/                                구 파이프라인
```

## 학습

저장소를 어디에 클론했든 `mochi_code/`가 기준입니다. `/home/dyan/nmf`는 예전 서버 경로라 로컬에 없어도 됩니다.

저장소 루트에서 가상환경을 만든 뒤 전처리 패키지만 깐다. torch CUDA 휠은 NVIDIA 머신에서만.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r ../requirements.txt
bash scripts/fetch_brca.sh
# NVIDIA GPU가 있을 때만 torch CUDA 휠을 깐 뒤:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
bash scripts/run_next_experiments.sh
```

단일 학습만 돌리려면:

```bash
python train_nmf_tf.py --gpu 0 --data_dir processed_data/gate_brca \
  --save_dir results/current/lr_brca_nogan_sp \
  --gamma_nonneg --w_head_act softplus --gan_to_mse 0
```

체크포인트: `results/current/lr_{brca,luad,kirc}_hybrid/nmf_tf_best.ckpt`  
그림: `paper/figures/fig1_pathway_knockout.pdf`
