"""Freeze the evaluation sets: validate them and record their fingerprints in MANIFEST.json.

Usage (inside WSL, conda env cmpe297, from the project folder):
    python src/freeze_eval_set.py                     # first freeze
    python src/freeze_eval_set.py --verify            # check files still match the manifest
    python src/freeze_eval_set.py --refreeze --reason "why the set had to change"

Freezing re-runs every structural and label check (including that all image files exist)
and refuses to freeze an invalid set. MANIFEST.json then records, per file: SHA-256, row
and image counts, yes/no counts per split, the builder settings from its .meta.json, and
the fingerprints of the source annotations (and the GQA vocabulary).

Nothing is locked: the manifest makes changes detectable. A deliberate change after
freezing needs --refreeze with a reason; the previous entry is kept under "history" so the
report can say what changed and why. score.py calls verify() and refuses to score against
a set that no longer matches.
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from build_pope_coco import K as K_COCO, load_coco, sha256
from build_pope_gqa import K as K_GQA, load_gqa
from pope_sampling import SPLITS, validate_rows

PROJECT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT / "data/eval_sets"
MANIFEST = EVAL_DIR / "MANIFEST.json"
SETS = ["coco_pope.jsonl", "gqa_pope.jsonl"]


def load_rules(dataset: str, data_root: str):
    """(k, present, negative_pool) from the same source data the builder used."""
    if dataset == "coco":
        _, vocab, present, _ = load_coco(os.path.join(data_root, "coco/annotations/instances_val2017.json"))
        return K_COCO, present, {f: set(vocab) - set(objs) for f, objs in present.items()}
    _, present, pool, _ = load_gqa(os.path.join(data_root, "gqa/sceneGraphs/val_sceneGraphs.json"))
    return K_GQA, present, pool


def describe(name: str, data_root: str) -> dict:
    """Validate one eval set and return its manifest entry; exit on any problem."""
    path = EVAL_DIR / name
    rows = [json.loads(line) for line in open(path) if line.strip()]
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    digest = sha256(path)
    problems = []
    if meta["output_sha256"] != digest:
        problems.append(f"file differs from what the builder wrote ({meta['builder']}); rebuild it")
    dataset = rows[0]["dataset"]
    k, present, pool = load_rules(dataset, data_root)
    images = list(dict.fromkeys(r["image"] for r in rows))
    problems += validate_rows(rows, len(images), k, present, pool, data_root=data_root)
    if problems:
        print(f"{name}: NOT FROZEN, {len(problems)} problem(s):", *problems[:20], sep="\n  ")
        sys.exit(1)
    counts = Counter((r["category"], r["answer"]) for r in rows)
    keep = ["builder", "built_utc", "seed", "k_per_image", "images_from", "n_eligible_images",
            "n_vocab_objects", "annotation_file", "annotation_sha256", "scene_graph_file",
            "scene_graph_sha256", "vocab_file", "vocab_sha256"]
    return {
        "sha256": digest,
        "dataset": dataset,
        "rows": len(rows),
        "images": len(images),
        "counts": {f"{s}/{a}": counts[s, a] for s in SPLITS for a in ("yes", "no")},
        "build": {key: meta[key] for key in keep if meta.get(key) is not None},
    }


def verify(manifest_path: Path = MANIFEST) -> list[str]:
    """Return problems (empty = every frozen file matches its recorded SHA-256)."""
    if not manifest_path.exists():
        return [f"{manifest_path} does not exist; run src/freeze_eval_set.py"]
    manifest = json.loads(manifest_path.read_text())
    problems = []
    for name, entry in manifest["files"].items():
        path = manifest_path.parent / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif sha256(path) != entry["sha256"]:
            problems.append(f"{name}: SHA-256 differs from MANIFEST.json (file changed after freezing)")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.environ.get("CMPE297_DATA", os.path.expanduser("~/cmpe297/data")))
    ap.add_argument("--verify", action="store_true", help="only check files against the manifest")
    ap.add_argument("--refreeze", action="store_true", help="replace an existing freeze (needs --reason)")
    ap.add_argument("--reason", help="why the frozen sets had to change (recorded in the manifest)")
    args = ap.parse_args()

    if args.verify:
        problems = verify()
        if problems:
            print("VERIFY FAILED:", *problems, sep="\n  ")
            sys.exit(1)
        files = json.loads(MANIFEST.read_text())["files"]
        print(f"verified: {', '.join(files)} match MANIFEST.json")
        return

    history = []
    if MANIFEST.exists():
        if not (args.refreeze and args.reason):
            sys.exit(f"{MANIFEST.name} already exists. The sets are frozen; to change them deliberately "
                     "use --refreeze --reason \"...\" (the old entry is kept in the history).")
        old = json.loads(MANIFEST.read_text())
        history = old.pop("history", []) + [old]

    manifest = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": args.reason or "initial freeze",
        "files": {name: describe(name, args.data_root) for name in SETS},
        "history": history,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")
    for name, entry in manifest["files"].items():
        print(f"frozen {name}: {entry['rows']:,} rows, {entry['images']} images, sha256 {entry['sha256'][:16]}...")
    print(f"wrote {MANIFEST.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
