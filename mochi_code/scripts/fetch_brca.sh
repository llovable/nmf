#!/usr/bin/env bash
# BRCA 원자료를 Xena에서 받아 전처리한다. 이미 있으면 건너뛴다.
# 저장소 어디에 클론했든 이 스크립트 위치 기준으로 경로를 잡는다.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$CODE/.." && pwd)"
RAW="${RAW:-$REPO/raw_data}"
CLIN="${CLIN:-$CODE/processed_data/clinical}"
OUT="${OUT:-$CODE/processed_data/gate_brca}"
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

fetch "https://gdc.xenahubs.net/download/TCGA-BRCA.star_tpm.tsv.gz" "$RAW/TCGA-BRCA.star_tpm.tsv.gz"
fetch "https://gdc.xenahubs.net/download/TCGA-BRCA.protein.tsv.gz" "$RAW/TCGA-BRCA.protein.tsv.gz"
fetch "https://gdc.xenahubs.net/download/TCGA-BRCA.methylation450.tsv.gz" "$RAW/TCGA-BRCA.methylation450.tsv.gz"
fetch "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix" "$CLIN/BRCA_clinicalMatrix"
fetch "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/h.all.v2023.2.Hs.symbols.gmt" "$CLIN/hallmark.gmt"
fetch "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_22/gencode.v22.metadata.HGNC.gz" "$RAW/gencode.v22.metadata.HGNC.gz"

python3 - "$RAW" "$CLIN" <<'PY'
import gzip, sys
from pathlib import Path
raw, clin = Path(sys.argv[1]), Path(sys.argv[2])
src = raw / "gencode.v22.metadata.HGNC.gz"
dst = clin / "gencode.probeMap"
if src.exists() and not dst.exists():
    rows = ["id\tgene"]
    with gzip.open(src, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[0].startswith("ENSG"):
                rows.append(f"{p[0]}\t{p[1]}")
    dst.write_text("\n".join(rows) + "\n")
    print(f"wrote {dst} n={len(rows)-1}")
else:
    print("probeMap", "exists" if dst.exists() else "skip")
PY

echo "downloads ready at $RAW"
if [[ -f "$OUT/rna.train.tsv" ]]; then
  echo "skip preprocess ($OUT already has rna.train.tsv)"
else
  echo "preprocessing into $OUT"
  python3 "$CODE/prepare_gate_data.py" \
    --rna "$RAW/TCGA-BRCA.star_tpm.tsv.gz" \
    --methy "$RAW/TCGA-BRCA.methylation450.tsv.gz" \
    --protein "$RAW/TCGA-BRCA.protein.tsv.gz" \
    --out_dir "$OUT"
fi
echo "done"
