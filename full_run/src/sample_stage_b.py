"""Draw the Stage B error-review lists from the ordinary-prompt full runs.

Usage (inside WSL, conda env cmpe297, from the project folder):
    python src/sample_stage_b.py --seed 297

Per dataset, two files in data/audit/:
- audit_fp_<dataset>.csv: false positives, i.e. questions labelled "no" that the ordinary
  prompt answered "yes". All of them, or a seeded sample of --fp-cap if there are more.
- audit_fn_<dataset>.csv: a seeded sample of --fn-n false negatives (labelled "yes",
  answered "no").

Units are distinct (image, object) questions, as in Stage A: a positive occurs in all three
splits and some negatives in two; with identical image and prompt, greedy decoding gives the
same answer in each copy (checked here; the script stops if it does not). `splits` lists
where the question occurs. Rows are shuffled. Reviewers tag each item with the error
taxonomy in audit_tool.STAGE_B.
"""
import argparse
import csv
import random
import sys
from collections import defaultdict

from freeze_eval_set import verify
from pope_sampling import SPLITS
from results_io import DATASETS, PROJECT, load_predictions, load_questions, samples_file

AUDIT_DIR = PROJECT / "data/audit"
# `overlap` (always 0) mirrors the Stage A columns; kept so redraws match the shipped lists.
COLUMNS = ["audit_id", "dataset", "image", "object", "answer", "response", "splits", "question_ids",
           "size_bin", "freq_rank", "cooc_score", "overlap"]


def error_pairs(questions: list[dict], preds: dict, label: str, wrong: str) -> dict:
    """(image, object) -> questions with ground truth `label` that were answered `wrong`."""
    groups = defaultdict(list)
    for q in questions:
        if q["answer"] == label:
            groups[q["image"], q["object"]].append(q)
    errors = {}
    for pair, qs in groups.items():
        answers = {preds[q["question_id"]]["response"] for q in qs}
        if len(answers) > 1:
            sys.exit(f"{pair}: copies of one question got different responses {answers}")
        if preds[qs[0]["question_id"]]["prediction"] == wrong:
            errors[pair] = qs
    return errors


def to_rows(dataset: str, kind: str, picked: list, errors: dict, preds: dict) -> list[dict]:
    rows = []
    for n, pair in enumerate(picked, 1):
        qs = errors[pair]
        first = qs[0]
        rows.append({
            "audit_id": f"{dataset}_{kind}{n:03d}",
            "dataset": dataset, "image": pair[0], "object": pair[1], "answer": first["answer"],
            "response": preds[first["question_id"]]["response"].strip(),
            "splits": ";".join(s for s in SPLITS if any(q["category"] == s for q in qs)),
            "question_ids": ";".join(q["question_id"] for q in qs),
            "size_bin": first["size_bin"] or "",
            "freq_rank": first["freq_rank"],
            "cooc_score": "" if first["cooc_score"] is None else first["cooc_score"],
            "overlap": 0,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=297)
    ap.add_argument("--prompt", default="ordinary")
    ap.add_argument("--fp-cap", type=int, default=100)
    ap.add_argument("--fn-n", type=int, default=50)
    ap.add_argument("--force", action="store_true", help="overwrite existing lists")
    args = ap.parse_args()

    problems = verify()
    if problems:
        sys.exit("frozen sets do not match MANIFEST.json:\n  " + "\n  ".join(problems))
    for dataset in DATASETS:
        questions = load_questions(dataset)
        preds = load_predictions(dataset, args.prompt)
        print(f"{dataset}: responses from {samples_file(dataset, args.prompt).name}")
        for kind, label, wrong, n in (("fp", "no", "yes", args.fp_cap), ("fn", "yes", "no", args.fn_n)):
            out = AUDIT_DIR / f"audit_{kind}_{dataset}.csv"
            if out.exists() and not args.force:
                sys.exit(f"{out.relative_to(PROJECT)} exists; reviewer labels refer to it (use --force to redraw)")
            errors = error_pairs(questions, preds, label, wrong)
            rng = random.Random(f"{args.seed}:{dataset}:{kind}")
            pool = sorted(errors)
            picked = rng.sample(pool, n) if len(pool) > n else pool
            rng.shuffle(picked)
            rows = to_rows(dataset, kind, picked, errors, preds)
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, COLUMNS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            n_rows = sum(len(qs) for qs in errors.values())
            print(f"  wrote {out.relative_to(PROJECT)}: {len(rows)} of {len(pool)} distinct {kind.upper()} "
                  f"questions ({n_rows} rows across splits)")


if __name__ == "__main__":
    main()
