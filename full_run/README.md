# Can VLMs Reject Objects That Are Not Present?

CMPE 297 Term Project 5, Group 7.

We measure object hallucination in **Qwen2.5-VL-3B-Instruct** with POPE-style yes/no questions ("Is
there a dog in the image?"). The questions are asked on **COCO val2017** (baseline) and **GQA**
(challenge set).
- **Negatives:** random, popular and adversarial negative sampling.
- **Prompts:** three predeclared prompt formats (ordinary, neutral, and evidence-aware with an explicit
  "not sure" option).
- **Label audit:** a manual audit of the dataset labels.

Results:
- **Report:** [report/technical_report.md](report/technical_report.md) (PDF alongside).
- **Notebook:** [notebooks/pope_hallucination_analysis.ipynb](notebooks/pope_hallucination_analysis.ipynb),
  saved with outputs.

## Contents

```
README.md                       this file
requirements.txt                pinned environment (pip freeze of the conda env cmpe297)
setup/
  01_conda_env.sh               create conda env cmpe297, install deps, register Jupyter kernel
  02_download_data.sh           COCO val2017 images + annotations, GQA val scene graphs
  03_gate_check.sh              GPU + model load + two test generations (go / no-go)
src/
  pope_sampling.py              shared POPE sampler (random / popular / adversarial)
  build_pope_coco.py            builds data/eval_sets/coco_pope.jsonl
  build_pope_gqa.py             builds data/eval_sets/gqa_pope.jsonl
  gqa_vocab.py                  GQA 100-object vocabulary, synonym and blocker lists
  fetch_gqa_images.py           downloads only the 500 GQA images the eval set uses
  compare_with_teammate_set.py  consistency check against the archived original COCO question file
  freeze_eval_set.py            validates the eval sets and writes / verifies MANIFEST.json
  prompts.py                    the three prompt templates (single source of truth)
  parser.py                     fixed yes / no / abstain / invalid response parser
  make_task_yamls.py            renders the six lmms-eval task YAMLs with local paths
  run_all.sh                    full inference runs (6 tasks x 9,000 questions)
  results_io.py                 loads logged responses and re-parses them
  sample_audit.py               draws the Stage A audit sample (predeclared, blind)
  sample_stage_b.py             draws the Stage B model-error review lists
  audit_tool.py                 ipywidgets review tool used by notebooks/label_audit.ipynb
  score.py                      metrics, bootstrap CIs, split comparisons, audit views
  analyze.py                    effect studies (size, co-occurrence, wording, categories), figures
tasks/pope_custom/              lmms-eval task: utils.py + six generated YAMLs
tests/test_parser.py            parser, prompt and metric unit tests
data/
  eval_sets/                    frozen question sets + MANIFEST.json (SHA-256 of each file)
  audit/                        audit samples, Stage B lists, reviewer label files
results/
  raw/full/                     lmms-eval samples + results per task, run logs, run_info.jsonl
  raw/smoke/                    smoke-test outputs (20 questions per task)
  tables/                       every table the report uses (generated)
  figures/                      every figure the report uses (generated)
notebooks/
  pope_hallucination_analysis.ipynb  analysis notebook (outputs retained)
  label_audit.ipynb             reviewer UI for the label audit
report/                         technical_report.md / .pdf, report.css (PDF styling)
archive/pope_val2017.jsonl      original teammate question file (source of the 500 COCO images)
```

## What you can do without a GPU

`data/eval_sets/`, `data/audit/` and `results/raw/` are shipped, so every table can be re-scored
without running the model. Inside the environment (see Setup), from the project folder:

```bash
python -m pytest tests/              # 44 tests: parser, prompts, metric arithmetic
python src/freeze_eval_set.py --verify
python src/score.py                  # metrics, CIs, audit views          (~2 s)
python src/analyze.py                # effect studies and figures          (~10 s)
```

- `score.py` needs no images.
- `analyze.py` needs the images only for the two example-image figures. Set `CMPE297_DATA` to a
  folder with `coco/val2017/` and `gqa/images/` (see Data).

## Hardware and software used

| | |
|---|---|
| OS | Windows 11 + WSL2 Ubuntu 22.04 (all commands run inside WSL) |
| GPU | NVIDIA RTX 4080, 16 GB (the 3B model in bf16 fits comfortably; `03_gate_check.sh` prints peak VRAM) |
| Python | 3.11 (conda env `cmpe297`) |
| Key packages | torch 2.6.0+cu124, transformers 5.23.0, lmms-eval 0.7.4, qwen-vl-utils 0.0.14, accelerate 1.15.0 |
| Model | `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, bf16 |
| Decoding | greedy, `max_new_tokens` 16, batch size 1; lmms-eval `qwen2_5_vl` defaults: system prompt "You are a helpful assistant.", `min_pixels` 200,704, `max_pixels` 1,605,632 (no image is downscaled) |

The full version list is in [requirements.txt](requirements.txt).

## Setup

All commands run **inside WSL**. Keep the heavy data on the Linux filesystem, not under `/mnt/c`. The
project folder path contains spaces, so quote it:

```bash
export PROJ="/mnt/c/Users/<you>/Documents/MSAI Classes/CMPE 297 - Computer Vision/Project"
cd "$PROJ"
bash setup/01_conda_env.sh      # conda env cmpe297 (Python 3.11), deps, Jupyter kernel "cmpe297 (py311)"
conda activate cmpe297
bash setup/02_download_data.sh  # ~1.1 GB: COCO val2017 + annotations, GQA val scene graphs
bash setup/03_gate_check.sh     # first run downloads ~7.5 GB of model weights; must print GATE PASSED
```

- `01_conda_env.sh` installs the latest compatible packages.
- To reproduce the exact environment instead, run this inside the new env:
  `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`.

### Data location (`CMPE297_DATA`)

Every script reads the data root from the environment variable `CMPE297_DATA` (default
`~/cmpe297/data`). `01_conda_env.sh` stores it, together with `HF_HOME` (default
`~/cmpe297/hf_cache`), on the conda env, so both are set whenever `cmpe297` is active.

The eval sets store only dataset-relative image file names, so the data can live anywhere. To use
another location (for example a Windows drive):

```bash
conda env config vars set CMPE297_DATA=/mnt/d/CMPE297CV_Data -n cmpe297   # then re-activate the env
```

The expected layout under `CMPE297_DATA`:

```
coco/val2017/*.jpg                          5,000 images
coco/annotations/instances_val2017.json
gqa/sceneGraphs/val_sceneGraphs.json
gqa/images/*.jpg                            the 500 images the GQA eval set uses (fetch_gqa_images.py)
```

## Reproducing everything, in order

Run from the project folder with `cmpe297` active. Steps 1-3 rebuild the shipped frozen sets
byte-for-byte. You can also skip to step 4 and use the shipped sets.

| # | Command | What it does | Time |
|---|---|---|---|
| 1 | `python src/build_pope_coco.py --images-from archive/pope_val2017.jsonl` | COCO set on the original 500 images (seed 297) | < 1 min |
| | `python src/compare_with_teammate_set.py` | consistency table vs the original question file | < 1 min |
| 2 | `python src/build_pope_gqa.py` then `python src/fetch_gqa_images.py` | GQA set (seed 297), then its 500 images from Visual Genome | 1-2 min |
| 3 | `python src/freeze_eval_set.py` | validates both sets and writes `MANIFEST.json` (refuses if one exists; `--verify` checks it) | < 1 min |
| 4 | `python src/make_task_yamls.py` | renders the six task YAMLs with this machine's absolute paths (needed on every machine) | seconds |
| 5 | smoke test (below) | 20 questions per task | ~5 min |
| 6 | `bash src/run_all.sh` | 54,000 generations, logs to `results/raw/full/` | ~2.5 h |
| 7 | `python src/sample_audit.py --seed 297` | Stage A audit sample (shipped; refuses to overwrite without `--force`) | seconds |
| | `python src/sample_stage_b.py --seed 297` | Stage B error lists from the ordinary-prompt run (shipped) | seconds |
| | `notebooks/label_audit.ipynb` | manual review; writes `data/audit/labels_*_<reviewer>.csv` (shipped) | hours |
| 8 | `python src/score.py` | `results/tables/` metrics, CIs, split comparisons, audit views | ~2 s |
| 9 | `python src/analyze.py` | effect-study tables and `results/figures/` | ~10 s |
| 10 | notebook (below) | runs score + analyze and shows every result | ~1 min |

**Smoke test (step 5).** This was the gate before the full runs. It checks that samples files are
written and that there are no invalid parses:

```bash
for task in pope_{coco,gqa}_{ordinary,neutral,evidence}; do
  python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --include_path "$PROJ/tasks/pope_custom" --tasks "$task" --limit 20 \
    --batch_size 1 --log_samples --output_path "$PROJ/results/raw/smoke/"
done
```

**Full runs (step 6).** `run_all.sh` first verifies the manifest, re-renders the YAMLs and runs the
tests. It then runs each of the six tasks. A task whose samples file is already complete is skipped,
so an interrupted run can simply be restarted. Each finished task appends its versions and model
revision to `results/raw/full/run_info.jsonl`.

**Notebook (step 10).** Start Jupyter from the project folder, open
`notebooks/pope_hallucination_analysis.ipynb`, choose the kernel **cmpe297 (py311)**, then *Restart and
Run All*. Or, headless:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=cmpe297 notebooks/pope_hallucination_analysis.ipynb
```

**Report PDF.** The PDF was exported from the Markdown with pandoc and Edge (Windows PowerShell,
from the project folder). The VS Code Markdown PDF extension works too.

```powershell
pandoc report/technical_report.md -f gfm -t html5 --standalone --embed-resources --resource-path=report `
  --css report/report.css --metadata pagetitle="CMPE 297 Group 7 technical report" -o report.html
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless=new --no-pdf-header-footer `
  --print-to-pdf="$PWD\report\technical_report.pdf" "file:///$($PWD -replace '\\','/')/report.html"
```

## Notes on reproducibility

- **Frozen sets.** `data/eval_sets/*.jsonl` never change after freezing. Every script verifies them
  against `MANIFEST.json` and stops if a hash differs. Audit verdicts are stored separately and joined
  at scoring time.
- **Re-parsed scores.** Scores are always recomputed from the raw logged responses with
  `src/parser.py`, not taken from the values computed during the run.
- **Seeds and determinism.** All sampling uses seed 297: question sets, audit samples, Stage B lists
  and bootstrap resamples. Decoding is greedy with batch size 1. Re-running `score.py` and
  `analyze.py` produces identical tables.
- **Example images.** The examples in the figures are fixed by id in `src/analyze.py`
  (`EXAMPLES_FP`, `EXAMPLES_LABEL_ERROR`).
