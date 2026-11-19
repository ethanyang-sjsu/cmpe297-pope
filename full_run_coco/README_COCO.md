# Can VLMs Reject Objects That Are Not Present? (COCO)


We measure object hallucination in **Qwen2.5-VL-3B-Instruct** with POPE-style yes/no questions ("Is
there a dog in the image?") on **COCO val2017**.
- **Negatives:** random, popular and adversarial negative sampling.
- **Prompts:** three predeclared prompt formats (ordinary, neutral, and evidence-aware with an explicit
  "not sure" option).
- **Label audit:** a manual audit of the dataset labels.

Results:
- **Notebook:** [notebooks/pope_hallucination_analysis_coco.ipynb](notebooks/pope_hallucination_analysis_coco.ipynb),
  saved with outputs.

The workflow below needs only the COCO data. 

## Contents

```
README_COCO.md                  this file
requirements.txt                pinned environment (pip freeze of the conda env cmpe297)
setup/
  01_conda_env.sh               create conda env cmpe297, install deps, register Jupyter kernel
  02_download_data_coco.sh      COCO val2017 images + annotations
  03_gate_check.sh              GPU + model load + two test generations (go / no-go)
src/
  pope_sampling.py              POPE sampler (random / popular / adversarial)
  build_pope_coco.py            builds data/eval_sets/coco_pope.jsonl
  freeze_eval_set_coco.py       checks the set against MANIFEST.json (--validate re-runs all checks)
  prompts.py                    the three prompt templates (single source of truth)
  parser.py                     fixed yes / no / abstain / invalid response parser
  make_task_yamls_coco.py       renders the three COCO lmms-eval task YAMLs with local paths
  run_all_coco.sh               full inference runs (3 tasks x 9,000 questions)
  results_io.py                 loads logged responses and re-parses them
  sample_audit_coco.py          draws the Stage A audit sample (predeclared, blind)
  sample_stage_b_coco.py        draws the Stage B model-error review lists
  audit_tool.py                 ipywidgets review tool used by notebooks/label_audit_coco.ipynb
  score_coco.py                 metrics, bootstrap CIs, split comparisons, audit views
  analyze_coco.py               effect studies (size, co-occurrence, wording, categories), figures
tasks/pope_custom/              lmms-eval task: utils.py + generated YAMLs (pope_coco_<prompt>.yaml)
tests/test_coco.py              parser, prompt and metric unit tests
data/
  eval_sets/                    frozen question set coco_pope.jsonl + MANIFEST.json (SHA-256)
  audit/                        audit sample, Stage B lists, reviewer label files (*_coco*.csv)
results/
  raw/full/                     lmms-eval samples & results per task
  raw/smoke/                    smoke-test outputs (20 questions per task)
  tables/coco/                  every table the report uses (generated)
  figures/coco/                 every figure the report uses (generated)
notebooks/
  pope_hallucination_analysis_coco.ipynb  analysis notebook
  label_audit_coco.ipynb        reviewer UI for the label audit
archive/                        original question file
```

## Setup

The full version list is in [requirements.txt](requirements.txt).

All commands run **inside WSL**. Keep the heavy data on the Linux filesystem, not under `/mnt/c`. The
project folder path contains spaces, so quote it:

```bash
export PROJ="/mnt/c/Users/<you>/Documents/MSAI Classes/CMPE 297 - Computer Vision/Project"
cd "$PROJ"
bash setup/01_conda_env.sh            # conda env cmpe297 (Python 3.11), deps, Jupyter kernel "cmpe297 (py311)"
conda activate cmpe297
bash setup/02_download_data_coco.sh   # ~1.1 GB: COCO val2017 images + annotations
bash setup/03_gate_check.sh           # first run downloads ~7.5 GB of model weights; must print GATE PASSED
```

- `01_conda_env.sh` installs the latest compatible packages.
- To reproduce the exact environment instead, run this inside the new env:
  `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`.

### Data location (`CMPE297_DATA`)

Every script reads the data root from the environment variable `CMPE297_DATA` (default
`~/cmpe297/data`). `01_conda_env.sh` stores it, together with `HF_HOME` (default
`~/cmpe297/hf_cache`), on the conda env, so both are set whenever `cmpe297` is active.

The eval set stores only dataset-relative image file names, so the data can live anywhere. To use
another location (for example a Windows drive):

```bash
conda env config vars set CMPE297_DATA=/mnt/d/CMPE297CV_Data -n cmpe297   # then re-activate the env
```

The expected layout under `CMPE297_DATA`:

```
coco/val2017/*.jpg                          5,000 images
coco/annotations/instances_val2017.json
```

## Reproducing everything, in order

Run from the project folder with `cmpe297` active. Step 1 rebuilds the shipped COCO set
byte-for-byte; the set is already frozen in `MANIFEST.json`, so step 2 checks it against that entry.
You can also skip to step 3 and use the shipped set.

| # | Command | What it does | Time |
|---|---|---|---|
| 1 | `python src/build_pope_coco.py --images-from archive/pope_val2017.jsonl` | COCO set on the original 500 images (seed 297) | < 1 min |
| | `python src/compare_with_teammate_set.py` | consistency table vs the original question file | < 1 min |
| 2 | `python src/freeze_eval_set_coco.py --validate` | SHA-256 check against `MANIFEST.json`, then every structural and label check (needs the images) | < 1 min |
| 3 | `python src/make_task_yamls_coco.py` | renders the three task YAMLs with this machine's absolute paths (needed on every machine) | seconds |
| 4 | smoke test (below) | 20 questions per task | ~3 min |
| 5 | `bash src/run_all_coco.sh` | 27,000 generations, logs to `results/raw/full/` | ~1.2 h |
| 6 | `python src/sample_audit_coco.py --seed 297` | Stage A audit sample (shipped; refuses to overwrite without `--force`) | seconds |
| | `python src/sample_stage_b_coco.py --seed 297` | Stage B error lists from the ordinary-prompt run (shipped) | seconds |
| | `notebooks/label_audit_coco.ipynb` | manual review; writes `data/audit/labels_*coco_<reviewer>.csv` (shipped) | hours |
| 7 | `python src/score_coco.py` | `results/tables/coco/` metrics, CIs, split comparisons, audit views | ~2 s |
| 8 | `python src/analyze_coco.py` | effect-study tables and `results/figures/coco/` | ~10 s |
| 9 | notebook (below) | runs steps 7-8 and shows every result | ~1 min |

**Smoke test (step 4).** This was the gate before the full runs. It checks that samples files are
written and that there are no invalid parses:

```bash
for task in pope_coco_{ordinary,neutral,evidence}; do
  python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --include_path "$PROJ/tasks/pope_custom" --tasks "$task" --limit 20 \
    --batch_size 1 --log_samples --output_path "$PROJ/results/raw/smoke/"
done
```

**Full runs (step 5).** `run_all_coco.sh` first verifies the set against the manifest, re-renders the
YAMLs and runs `tests/test_coco.py`. It then runs the three tasks (`pope_coco_ordinary`,
`pope_coco_neutral`, `pope_coco_evidence`). A task whose samples file is already complete is skipped,
so an interrupted run can simply be restarted. Each finished task appends its versions and model
revision to `results/raw/full/run_info.jsonl`. The three runs took 1 h 10 min on the RTX 4080.

**Notebook (step 9).** Start Jupyter from the project folder, open
`notebooks/pope_hallucination_analysis_coco.ipynb`, choose the kernel **cmpe297 (py311)**, then
*Restart and Run All*. Or, headless:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=cmpe297 notebooks/pope_hallucination_analysis_coco.ipynb
```

## Notes on reproducibility

- **Frozen set.** `data/eval_sets/coco_pope.jsonl` never changes after freezing. Every COCO script
  verifies it against its `MANIFEST.json` entry and stops if the hash differs. Audit verdicts are
  stored separately and joined at scoring time.
- **Re-parsed scores.** Scores are always recomputed from the raw logged responses with
  `src/parser.py`, not taken from the values computed during the run.
- **Seeds and determinism.** All sampling uses seed 297: question set, audit sample, Stage B lists and
  bootstrap resamples. Decoding is greedy with batch size 1. Re-running `score_coco.py` and
  `analyze_coco.py` produces identical tables.
- **Audit consistency.** `score_coco.py` stops if a Stage B item's *label error* tag disagrees with
  its verdict.
- **Example images.** The examples in the figures are fixed by id in `src/analyze_coco.py`
  (`EXAMPLES_FP`, `EXAMPLES_LABEL_ERROR`).
