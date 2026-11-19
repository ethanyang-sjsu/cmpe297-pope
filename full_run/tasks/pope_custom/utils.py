"""lmms-eval task functions for the frozen POPE sets (adapted from a teammate's custom
task, which was cut from lmms-eval's tasks/pope/utils.py).

Changes from the original:
- images are opened from $CMPE297_DATA (rows store only the dataset and file name);
- prompts come from src/prompts.py (built from the `object` field), one function per format;
- responses are classified by the fixed parser in src/parser.py instead of exact string
  match, so "Yes." counts as yes and abstentions are counted;
- extra in-run metrics: specificity, hallucination rate, abstention rate.

The aggregates printed by lmms-eval are a quick readout only. Official numbers come from
src/score.py, which re-parses the logged raw responses and adds confidence intervals.
"""
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from parser import parse  # noqa: E402
from pope_sampling import image_path  # noqa: E402
from prompts import build_prompt  # noqa: E402

DATA_ROOT = os.environ.get("CMPE297_DATA", os.path.expanduser("~/cmpe297/data"))
METRICS = ["pope_accuracy", "pope_precision", "pope_recall", "pope_f1_score", "pope_specificity",
           "pope_hallucination_rate", "pope_abstention_rate", "pope_yes_ratio"]


def pope_doc_to_visual(doc):
    return [Image.open(image_path(DATA_ROOT, doc["dataset"], doc["image"])).convert("RGB")]


def doc_to_text_ordinary(doc, lmms_eval_specific_kwargs=None):
    return build_prompt("ordinary", doc["object"])


def doc_to_text_neutral(doc, lmms_eval_specific_kwargs=None):
    return build_prompt("neutral", doc["object"])


def doc_to_text_evidence(doc, lmms_eval_specific_kwargs=None):
    return build_prompt("evidence", doc["object"])


def pope_process_results(doc, results):
    record = {"question_id": doc["question_id"], "response": results[0],
              "prediction": parse(results[0]), "ground_truth": doc["answer"]}
    return {m: record for m in METRICS}


def _counts(results):
    c = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "abstain": 0, "pos": 0, "neg": 0}
    for r in results:
        pred, gt = r["prediction"], r["ground_truth"]
        c["pos" if gt == "yes" else "neg"] += 1
        if pred in ("abstain", "invalid"):
            c["abstain"] += 1
        elif gt == "yes":
            c["tp" if pred == "yes" else "fn"] += 1
        else:
            c["fp" if pred == "yes" else "tn"] += 1
    return c


def _div(a, b):
    return a / b if b else 0.0


def pope_aggregate_accuracy(results):
    c = _counts(results)
    return _div(c["tp"] + c["tn"], len(results))


def pope_aggregate_precision(results):
    c = _counts(results)
    return _div(c["tp"], c["tp"] + c["fp"])


def pope_aggregate_recall(results):
    c = _counts(results)
    return _div(c["tp"], c["pos"])


def pope_aggregate_f1_score(results):
    p, r = pope_aggregate_precision(results), pope_aggregate_recall(results)
    return _div(2 * p * r, p + r)


def pope_aggregate_specificity(results):
    c = _counts(results)
    return _div(c["tn"], c["neg"])


def pope_aggregate_hallucination_rate(results):
    c = _counts(results)
    return _div(c["fp"], c["neg"])


def pope_aggregate_abstention_rate(results):
    return _div(_counts(results)["abstain"], len(results))


def pope_aggregate_yes_ratio(results):
    return _div(sum(r["prediction"] == "yes" for r in results), len(results))
