## 범용 멀티오믹스 모델 (샘플 수 가변 지원)

`mochi_code`는 기존 코드 그대로 두고,  
새 폴더에서 **샘플 수가 달라도 동작**하도록 만든 범용 버전입니다.

핵심 특징
- 샘플 수 제한 없음
- 입력 특징 수는 학습 시점에 자동 결정
- 새 데이터 입력 시 **특징 정렬/샘플 정렬 자동**

### 입력 데이터 형식 (중요)
- **TSV 형식, 행=특징 / 열=샘플 ID**
- 예: 유전자/프로브/단백질이 행, 환자 ID가 열
- 샘플 수는 달라도 됩니다.  
  다만 **특징 집합(개수/이름/순서)은 학습 기준으로 맞춰야** 합니다.

### 1) 학습 (Tri-joint)
```bash
python train_tri_joint_universal.py \
  --rna /path/to/rna.tsv \
  --methy /path/to/methy.tsv \
  --protein /path/to/protein.tsv \
  --output_dir ./results_universal \
  --use_attention \
  --num_epochs 100 \
  --batch_size 32
```

옵션:
- `--sample_id_mode`: `none` | `tcga12` | `tcga15`  
  (TCGA ID를 12/15자리로 통일해서 샘플 매칭)
- `--sample_join`: `intersection` | `union`  
  (모달리티 간 공통 샘플만 쓸지, 합집합으로 쓸지)

### 2) Imputation
```bash
python impute_universal.py \
  --checkpoint ./results_universal/tri_best.ckpt \
  --rna /path/to/rna.tsv \
  --methy /path/to/methy.tsv \
  --protein /path/to/protein.tsv \
  --output_dir ./impute_out
```

### 동작 방식 요약
- 학습 시점의 **특징 리스트를 checkpoint에 저장**
- Imputation 시 **특징을 자동 정렬**
  - 없는 특징은 NaN으로 추가
  - 추가 특징은 자동 제거
- 샘플은 **합집합 기준으로 자동 정렬**

> 주의: 학습 때와 **완전히 다른 특징 집합**이면 예측 품질이 낮아질 수 있습니다.

### 자주 묻는 질문
**Q. 다른 결측 모델도 행/열이 달라도 되나요?**  
A. 대부분의 결측 보간 모델은 **특징(열) 구조가 고정된 행렬**을 전제로 합니다.  
샘플 수는 가변이어도 되지만, **특징 집합은 동일해야** 정상 동작합니다.

**Q. 샘플×특징 형태로 되어 있는데요?**  
A. 반드시 **특징×샘플** 형태로 저장해주세요.  
즉, **행=특징 / 열=샘플 ID** 구조가 기준입니다.

### 추가 결측 모델(선택형) 제안
아래 모델을 **사용자가 선택해서 실행**할 수 있는 방식으로 확장하는 것을 권장합니다.
- **Mean/Median Imputation**: 가장 단순한 기준선, 빠르고 안정적
- **KNN Imputation**: 유사 샘플 기반 보간 (scikit-learn)
- **SoftImpute (Matrix Completion)**: 저랭크 가정 기반
- **MissForest**: 비선형 랜덤포레스트 기반

> 위 모델들은 모두 **특징(열) 구조가 고정된 행렬**을 전제로 합니다.

### 추가 모델 모듈 위치
아래 모듈들이 실제 구현으로 추가되어 있습니다.
- `imputers/mean_median.py` → `impute_mean`, `impute_median`
- `imputers/knn.py` → `impute_knn`
- `imputers/softimpute.py` → `impute_softimpute` (fancyimpute 필요)
- `imputers/missforest.py` → `impute_missforest` (missingpy 필요)

필요한 패키지가 없으면 `ImportError`로 안내됩니다.

### 모델 선택 실행 스크립트
```bash
python run_imputer.py \
  --input /path/to/input.tsv \
  --output /path/to/output.tsv \
  --model knn \
  --params '{"n_neighbors": 5}'
```

모델 목록 확인:
```bash
python run_imputer.py --list_models
```

파라미터 JSON 파일로 전달:
```bash
python run_imputer.py \
  --input /path/to/input.tsv \
  --output /path/to/output.tsv \
  --model missforest \
  --params_file /path/to/params.json
```
