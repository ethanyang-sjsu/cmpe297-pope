"""Draw the COCO Stage A label-audit sample from the frozen evaluation set.

Usage (inside WSL, conda env cmpe297, from the project folder):
    python src/sample_audit_coco.py --seed 297

COCO-only version of sample_audit.py (same draw, so the same sample for the same seed):
150 positives and 150 negatives (50 from each of random, popular, adversarial), written to
data/audit/audit_sample_coco.csv. The draw depends only on the frozen set and the seed, never
on model outputs, so it is predeclared.

- Positives are the same (image, object) questions in all three splits, so they are drawn
  from the 1,500 distinct positive pairs. Negatives are drawn per split; a pair that already
  appears in the sample (popular and adversarial can pick the same object) is not drawn again.
- `drawn_from` is "positive" or the split a negative was drawn for (the stratum used for
  per-split label-noise estimates).
- `splits` lists every split in which the (image, object) question occurs; score_coco.py joins
  audit verdicts to questions on (image, object), so one verdict covers all of them.
- Rows are shuffled so reviewers see yes and no questions mixed; the first --overlap rows
  (a random subset) are marked for the second reviewer.
- Refuses to overwrite an existing sample (labels refer to its audit_ids) unless --force.
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from freeze_eval_set_coco import EVAL_DIR, verify
from pope_sampling import SPLITS

PROJECT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT / "data/audit"
DS = "coco"
N_POS = 150
N_NEG_PER_SPLIT = 50
COLUMNS = ["audit_id", "dataset", "image", "object", "answer", "drawn_from", "splits", "question_ids",
           "size_bin", "overlap"]


def draw(rows: list[dict], seed: int, overlap: int) -> list[dict]:
    rng = random.Random(f"{seed}:{DS}")
    by_pair = defaultdict(list)  # (image, object, answer) -> rows in every split
    for r in rows:
        by_pair[r["image"], r["object"], r["answer"]].append(r)

    positives = sorted(p for p in by_pair if p[2] == "yes")
    picked = [(p, "positive") for p in rng.sample(positives, N_POS)]
    for split in SPLITS:
        taken = {p for p, _ in picked}
        pool = sorted({(r["image"], r["object"], r["answer"]) for r in rows
                       if r["category"] == split and r["answer"] == "no"} - taken)
        picked += [(p, split) for p in rng.sample(pool, N_NEG_PER_SPLIT)]
    rng.shuffle(picked)

    sample = []
    for n, (pair, drawn_from) in enumerate(picked, 1):
        qs = by_pair[pair]
        sample.append({
            "audit_id": f"{DS}_a{n:03d}",
            "dataset": DS,
            "image": pair[0],
            "object": pair[1],
            "answer": pair[2],
            "drawn_from": drawn_from,
            "splits": ";".join(s for s in SPLITS if any(q["category"] == s for q in qs)),
            "question_ids": ";".join(q["question_id"] for q in qs),
            "size_bin": qs[0]["size_bin"] or "",
            "overlap": int(n <= overlap),
        })
    return sample


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=297)
    ap.add_argument("--overlap", type=int, default=50, help="items for the second reviewer")
    ap.add_argument("--force", action="store_true", help="overwrite an existing sample")
    args = ap.parse_args()

    problems = verify()
    if problems:
        sys.exit("frozen set does not match MANIFEST.json:\n  " + "\n  ".join(problems))
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIT_DIR / f"audit_sample_{DS}.csv"
    if out.exists() and not args.force:
        sys.exit(f"{out.relative_to(PROJECT)} exists; reviewer labels refer to it (use --force to redraw)")
    rows = [json.loads(line) for line in open(EVAL_DIR / f"{DS}_pope.jsonl")]
    sample = draw(rows, args.seed, args.overlap)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sample)
    n_yes = sum(s["answer"] == "yes" for s in sample)
    drawn = {s: sum(x["drawn_from"] == s for x in sample) for s in SPLITS}
    print(f"wrote {out.relative_to(PROJECT)}: {len(sample)} items ({n_yes} yes, {len(sample) - n_yes} no; "
          f"no drawn per split {drawn}), {len({s['image'] for s in sample})} images, "
          f"{args.overlap} marked for overlap")


if __name__ == "__main__":
    main()
