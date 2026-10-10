#!/usr/bin/env bash
# TCGA 코호트 원자료를 Xena에서 받는다. 이미 있으면 건너뛴다.
# 사용: fetch_xena_cohort.sh BRCA|LUAD|KIRC
set -euo pipefail
COHORT="${1:?usage: $0 BRCA|LUAD|KIRC}"
# 클론 루트. /workspace 에 묶으면 다른 기계에서는 생존 파일이 평가 경로 밖에 떨어진다.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RAW="$ROOT/raw_data"
CLIN="$ROOT/mochi_code/processed_data/clinical"
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
mkdir -p "$RAW" "$CLIN"

fetch() {
  local url="$1" dest="$2"
  if [[ -f "$dest" && -s "$dest" ]]; then
    echo "skip $dest"
    return
  fi
  echo "GET $url"
  curl -fL --retry 5 --retry-delay 8 -A "$UA" -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
  echo "saved $dest ($(du -h "$dest" | cut -f1))"
}

fetch "https://gdc.xenahubs.net/download/TCGA-${COHORT}.star_tpm.tsv.gz" \
  "$RAW/TCGA-${COHORT}.star_tpm.tsv.gz"
fetch "https://gdc.xenahubs.net/download/TCGA-${COHORT}.protein.tsv.gz" \
  "$RAW/TCGA-${COHORT}.protein.tsv.gz"
fetch "https://gdc.xenahubs.net/download/TCGA-${COHORT}.methylation450.tsv.gz" \
  "$RAW/TCGA-${COHORT}.methylation450.tsv.gz"
fetch "https://tcga.xenahubs.net/download/TCGA.${COHORT}.sampleMap/${COHORT}_clinicalMatrix" \
  "$CLIN/${COHORT}_clinicalMatrix"
# 생존 평가(eval_survival.py)에 필요하다. 임상 행렬과 다른 파일이다.
fetch "https://gdc.xenahubs.net/download/TCGA-${COHORT}.survival.tsv.gz" \
  "$CLIN/TCGA-${COHORT}.survival.tsv.gz"

echo "${COHORT} downloads ready"
