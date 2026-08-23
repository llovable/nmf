#!/usr/bin/env bash
# BRCA 원자료를 Xena에서 받아 전처리한다. 이미 있으면 건너뛴다.
set -euo pipefail
ROOT=/workspace
RAW="$ROOT/raw_data"
CLIN="$ROOT/mochi_code/processed_data/clinical"
OUT="$ROOT/mochi_code/processed_data/gate_brca"
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

python3 - <<'PY'
from pathlib import Path
import gzip
clin = Path("/workspace/mochi_code/processed_data/clinical")
src = Path("/workspace/raw_data/gencode.v22.metadata.HGNC.gz")
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

echo "downloads ready"
