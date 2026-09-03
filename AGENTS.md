# AGENTS.md

## Cursor Cloud specific instructions

MOCHI is a **research codebase** (Python-only) for imputing missing values in TCGA multi-omics
data. There is **no server/app, no database, and no test/lint/build tooling** — everything is a
batch-mode CLI Python script. The reported model is the NMF-Transformer (`mochi_code/train_nmf_tf.py`).
Per the README, only edit inside `mochi_code/`.

### Environment / dependencies
- Dependencies are installed into the **system Python** (`/usr/bin/python3`, 3.12) via
  `pip install --break-system-packages` (see the startup update script). There is **no venv** —
  just run `python3 ...`. The base image lacks `ensurepip`, so `python3 -m venv` fails unless you
  first `apt install python3.12-venv`; prefer system Python instead.
- `mochi_code/requirements.txt` lists `torch-linalg`, which **does not exist on PyPI** and is
  **not imported** by active code (the code uses the built-in `torch.linalg`). The update script
  filters that line out; if you install manually, exclude it or `pip install` will fail.
- The VM is **CPU-only** (`torch.cuda.is_available()` is `False`). Scripts take `--gpu 0` but fall
  back to CPU automatically (`train_nmf_tf.py`), so pass `--gpu 0` and expect CPU execution.

### Running the pipeline (non-obvious gotchas)
- Script default paths point at the original author's machine (`/home/dyan/...`) and **do not exist
  here**. Always pass `--data_dir` and `--save_dir` explicitly.
- The real TCGA inputs and processed splits are **private and gitignored** (`data/`,
  `mochi_code/processed_data/`, `mochi_code/results/`, `baselines/`), so they are absent. Without
  them you cannot reproduce paper numbers; you can still exercise the full pipeline on synthetic data.
- `TripleSplitDataset` (`mochi_code/train_gate.py`) expects, under `--data_dir`, TSV matrices named
  `{rna,protein,methy}.{train,val,test}.tsv`, laid out **features × samples** (feature IDs in
  column 0, sample IDs as the header row). It transposes internally and intersects sample IDs across
  the three modalities, z-scores using train stats, and treats NaNs as missing cells.
- To smoke-test end-to-end without real data: generate small synthetic splits in that format, then
  `cd mochi_code && python3 train_nmf_tf.py --data_dir <dir> --save_dir <out> --gpu 0 --k 6 --d_model 32 --n_layers 1 --epochs1 3 --epochs2 8 --batch_size 16`.
  Use small `--k`/`--d_model`/`--epochs*` — defaults (k=20, d_model=128, epochs2=150) are sized for
  the real cohorts. Training writes `nmf_tf_best.ckpt`; load it with `load_nmf_tf` and impute a
  fully-missing modality block via `predict_nmf_tf(model, tabs, device, missing=<modality>)`.
- Paper Figure 1 is self-contained: `python3 mochi_code/paper/figures/plot_pathway_knockout.py`
  reads the committed `source_pathway_knockout.csv` and rewrites `fig1_pathway_knockout.{pdf,png}`
  (tracked files — `git checkout` them afterward if you only meant to verify).
- The heavier `eval_*.py` scripts (e.g. `eval_nmf_tf.py`) additionally require external baseline
  checkpoints (MOCHI-v5, MIMIR, shared) that are not present; they are not runnable out of the box.
