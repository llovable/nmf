# 📚 Archive 폴더 안내

이 폴더에는 이전 버전의 코드와 문서가 보관되어 있습니다.

## 📁 구조

```
archive/
├── old_code/              # 이전 버전 코드 파일들
│   ├── enhanced_downstream_analysis_backup.py
│   ├── train_multi_omics.py
│   ├── run_multi_omics_pipeline.py
│   └── data_prep.py
│
└── old_readme/            # 이전 README 파일들
    ├── README_new_module.md
    └── README_new_modules.md
```

## 📝 파일 설명

### old_code/
- **enhanced_downstream_analysis_backup.py**: 이전 다운스트림 분석 모듈 (백업)
- **train_multi_omics.py**: 구버전 학습 스크립트 → 현재는 `multi_omics_train.py` 사용
- **run_multi_omics_pipeline.py**: 구버전 파이프라인 → 현재는 `run_complete_pipeline.py` 사용
- **data_prep.py**: 기본 데이터 전처리 → 현재는 `data_prep_enhanced.py` 사용

### old_readme/
- **README_new_module.md**: 다운스트림 분석 모듈 설명 (하나의 파일로 통합)
- **README_new_modules.md**: 다운스트림 분석 모듈 설명 (여러 모듈로 분리) - 실제 구현되지 않음

## ⚠️ 주의사항

이 파일들은 참고용으로만 보관되어 있으며, 현재 프로젝트에서는 사용되지 않습니다.
최신 코드는 상위 디렉토리의 파일들을 참조하세요.

