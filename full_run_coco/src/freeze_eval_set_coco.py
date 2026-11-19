"""Check the frozen COCO evaluation set against its MANIFEST.json entry.

Usage (inside WSL, conda env cmpe297, from the project folder):
    python src/freeze_eval_set_coco.py --verify     # SHA-256 check (no images needed)
    python src/freeze_eval_set_coco.py --validate   # also re-run every structural and label check

COCO-only version of freeze_eval_set.py. The COCO set is already frozen: its entry in
data/eval_sets/MANIFEST.json records the SHA-256, row and image counts, yes/no counts per split
and the builder settings. This script never writes the manifest; it only checks
coco_pope.jsonl against that entry. --validate additionally re-runs the builder's checks
(including that every image exists under CMPE297_DATA) and compares the counts with the entry.
score_coco.py and the other COCO scripts call verify() and refuse to run against a set that no
longer matches.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from build_pope_coco import K, load_coco, sha256
from pope_sampling import SPLITS, validate_rows

PROJECT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT / "data/eval_sets"
MANIFEST = EVAL_DIR / "MANIFEST.json"
SET = "coco_pope.jsonl"


def entry() -> dict:
    """The COCO entry of MANIFEST.json."""
    return json.loads(MANIFEST.read_text())["files"][SET]


def verify() -> list[str]:
    """Return problems (empty = coco_pope.jsonl matches its recorded SHA-256)."""
    if not MANIFEST.exists():
        return [f"{MANIFEST} does not exist"]
    if SET not in json.loads(MANIFEST.read_text())["files"]:
        return [f"{SET} has no entry in {MANIFEST.name}"]
    path = EVAL_DIR / SET
    if not path.exists():
        return [f"{SET}: missing"]
    if sha256(path) != entry()["sha256"]:
        return [f"{SET}: SHA-256 differs from MANIFEST.json (file changed after freezing)"]
    return []


def validate(data_root: str) -> list[str]:
    """Re-run the builder's structural and label checks and compare counts with the manifest entry."""
    path = EVAL_DIR / SET
    rows = [json.loads(line) for line in open(path) if line.strip()]
    _, vocab, present, _ = load_coco(os.path.join(data_root, "coco/annotations/instances_val2017.json"))
    pool = {f: set(vocab) - set(objs) for f, objs in present.items()}
    images = list(dict.fromkeys(r["image"] for r in rows))
    problems = validate_rows(rows, len(images), K, present, pool, data_root=data_root)
    e = entry()
    counts = Counter((r["category"], r["answer"]) for r in rows)
    if len(rows) != e["rows"] or len(images) != e["images"]:
        problems.append(f"{len(rows)} rows / {len(images)} images, manifest says {e['rows']} / {e['images']}")
    for s in SPLITS:
        for a in ("yes", "no"):
            if counts[s, a] != e["counts"][f"{s}/{a}"]:
                problems.append(f"{s}/{a}: {counts[s, a]} rows, manifest says {e['counts'][f'{s}/{a}']}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.environ.get("CMPE297_DATA", os.path.expanduser("~/cmpe297/data")))
    ap.add_argument("--verify", action="store_true", help="only check the file against the manifest (default)")
    ap.add_argument("--validate", action="store_true", help="also re-run the structural and label checks")
    args = ap.parse_args()

    problems = verify()
    if not problems and args.validate:
        problems = validate(args.data_root)
    if problems:
        print("VERIFY FAILED:", *problems[:20], sep="\n  ")
        sys.exit(1)
    e = entry()
    print(f"verified: {SET} matches MANIFEST.json ({e['rows']:,} rows, {e['images']} images, "
          f"sha256 {e['sha256'][:16]}...)" + ("; all structural and label checks pass" if args.validate else ""))


if __name__ == "__main__":
    main()
