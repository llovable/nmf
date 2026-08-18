# MOCHI

TCGA 멀티오믹스 **칸 결측**과 **블록 결측**을 한 모형에서 채우는 연구 코드입니다. OmicsNMF(NMF + GAN)를 세 오믹스 상호보정으로 확장합니다.

보고 모형은 `mochi_code/train_nmf_tf.py` (오토인코더 은닉 + 고정 NMF 토큰 + 저랭크 잔차 + 조건부 WGAN)입니다.

## 폴더

```
mochi_code/     현재 모델과 실험 (여기만 수정)
baselines/      공식 OmicsNMF, OmiTrans, MIMIR (비교군, 수정하지 않음, gitignore)
archive/        예전 파이프라인, 노트북, 툴킷
```

```bash
source /home/dyan/nmf/.nmf/bin/activate
cd mochi_code
python train_nmf_tf.py --gpu 0
python paper/figures/plot_pathway_knockout.py
```

그림 1은 `mochi_code/paper/figures/fig1_pathway_knockout.pdf`입니다.

## 커밋

- 모델·논문: `mochi_code/`만 스테이징 (결과·전처리 행렬은 gitignore)
- 공식 베이스라인은 `baselines/`에 그대로 둠
