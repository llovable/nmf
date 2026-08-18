# mochi_code

보고 모형은 NMF-Transformer(`train_nmf_tf.py`)입니다. 예전 NMF+TGAN·게이트 번역기는 `archive/`에 있습니다.

## 활성 파일

| 파일 | 역할 |
|---|---|
| `prepare_gate_data.py` | TCGA 종양 정렬, train/val/test |
| `models_nmf_tf.py` | 보고 모형: AE 은닉 + NMF 토큰 + 저랭크 잔차 |
| `train_nmf_tf.py` | **현재 학습** |
| `eval_biology.py` | 경로·환자구조·교차오믹스 보존 |
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

```bash
source /home/dyan/nmf/.nmf/bin/activate
python train_nmf_tf.py --gpu 0
python paper/figures/plot_pathway_knockout.py
```

체크포인트: `results/current/lr_{brca,luad,kirc}_hybrid/nmf_tf_best.ckpt`  
그림: `paper/figures/fig1_pathway_knockout.pdf`
