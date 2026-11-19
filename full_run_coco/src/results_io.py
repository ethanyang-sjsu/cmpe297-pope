"""Read the logged full-run outputs (shared by the Stage B sampler, score.py, analyze.py).

Each lmms-eval samples file holds, per question, the metric record written by
tasks/pope_custom/utils.py: question_id, raw response, in-run prediction, ground truth.
Predictions are always recomputed here from the raw response with the fixed parser, so
the official numbers never depend on what was computed during the run.
"""
import json
from pathlib import Path

from parser import parse

PROJECT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT / "results/raw/full"
EVAL_DIR = PROJECT / "data/eval_sets"
DATASETS = ["coco", "gqa"]
PROMPT_KINDS = ["ordinary", "neutral", "evidence"]


def samples_file(dataset: str, prompt: str) -> Path:
    """Newest samples file of task pope_<dataset>_<prompt> (the one run_all.sh checked)."""
    task = f"pope_{dataset}_{prompt}"
    files = sorted(RAW_DIR.glob(f"*/*_samples_{task}.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no samples file for {task} under {RAW_DIR}")
    return files[-1]


def load_questions(dataset: str) -> list[dict]:
    return [json.loads(line) for line in open(EVAL_DIR / f"{dataset}_pope.jsonl")]


def load_predictions(dataset: str, prompt: str) -> dict[str, dict]:
    """question_id -> {"response", "prediction"} for one task, re-parsed from raw responses."""
    out = {}
    for line in open(samples_file(dataset, prompt)):
        rec = json.loads(line)["pope_accuracy"]
        out[rec["question_id"]] = {"response": rec["response"], "prediction": parse(rec["response"])}
    return out
