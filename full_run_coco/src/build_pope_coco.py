"""Build the COCO val2017 POPE evaluation set (9,000 questions by default).

Usage (inside WSL, conda env cmpe297, from the project folder):
    python src/build_pope_coco.py --images-from archive/pope_val2017.jsonl
    python src/build_pope_coco.py                    # sample 500 new eligible images

Writes data/eval_sets/coco_pope.jsonl plus coco_pope.meta.json (build parameters and
summary). The output is validated before it is written; nothing is written on failure.

COCO-specific choices:
- "Present" means at least one instance annotation of the category, including
  iscrowd=1 annotations (as in POPE). An image's objects are ordered by first appearance
  in the annotation file; the first 3 are the positives.
- Negatives may be any of the 80 categories with no annotation in the image.
- An image is eligible if it has at least 3 distinct categories (needed for K=3 positives).
- Object size uses the largest instance's bounding box: obj_area_frac is its share of the
  image area; size_bin uses COCO's absolute thresholds (small < 32^2 px, medium < 96^2 px).
- Frequency and co-occurrence statistics come from the selected images (as in POPE).
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pope_sampling import SPLITS, ObjectStats, build_rows, validate_rows

PROJECT = Path(__file__).resolve().parents[1]
K = 3


def size_bin(area: float) -> str:
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def load_coco(ann_path: str):
    coco = json.load(open(ann_path))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["file_name"]: im for im in coco["images"]}
    id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
    present = defaultdict(list)
    largest = {}
    for a in coco["annotations"]:
        f, name = id_to_file[a["image_id"]], cat_name[a["category_id"]]
        if name not in present[f]:
            present[f].append(name)
        w, h = a["bbox"][2:]
        largest[f, name] = max(largest.get((f, name), 0.0), w * h)
    size_info = {(f, name): (area / (images[f]["width"] * images[f]["height"]), size_bin(area))
                 for (f, name), area in largest.items()}
    return images, sorted(cat_name.values()), dict(present), size_info


def images_from_jsonl(path: str) -> list[str]:
    with open(path) as fh:
        return list(dict.fromkeys(json.loads(line)["image"] for line in fh if line.strip()))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.environ.get("CMPE297_DATA", os.path.expanduser("~/cmpe297/data")))
    ap.add_argument("--images-from", help="reuse the image list (in order) of an existing POPE jsonl")
    ap.add_argument("--n-images", type=int, default=500, help="images to sample when --images-from is not given")
    ap.add_argument("--seed", type=int, default=297)
    ap.add_argument("--out", default=str(PROJECT / "data/eval_sets/coco_pope.jsonl"))
    args = ap.parse_args()

    ann_path = os.path.join(args.data_root, "coco/annotations/instances_val2017.json")
    images, vocab, present, size_info = load_coco(ann_path)
    eligible = sorted(f for f, objs in present.items() if len(objs) >= K)

    if args.images_from:
        selected = images_from_jsonl(args.images_from)
        bad = [f for f in selected if f not in present or len(present[f]) < K]
        if bad:
            sys.exit(f"{len(bad)} images from {args.images_from} are not eligible, e.g. {bad[:3]}")
    else:
        selected = random.Random(args.seed).sample(eligible, args.n_images)

    stats = ObjectStats({f: present[f] for f in selected}, vocab)
    negative_pool = {f: set(vocab) - set(present[f]) for f in selected}
    rows = build_rows("coco", selected, {f: images[f]["id"] for f in selected}, present,
                      negative_pool, size_info, stats, K, args.seed)

    errors = validate_rows(rows, len(selected), K, present, negative_pool, data_root=args.data_root)
    if errors:
        print("VALIDATION FAILED:", *errors[:20], sep="\n  ")
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    counts = Counter((r["category"], r["answer"]) for r in rows)
    meta = {
        "dataset": "coco",
        "output": out.name,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": "src/build_pope_coco.py",
        "seed": args.seed,
        "k_per_image": K,
        "images_from": os.path.relpath(args.images_from, PROJECT) if args.images_from else None,
        "n_images": len(selected),
        "n_eligible_images": len(eligible),
        "n_rows": len(rows),
        "counts": {f"{s}/{a}": counts[s, a] for s in SPLITS for a in ("yes", "no")},
        "positive_size_bins": dict(Counter(r["size_bin"] for r in rows
                                           if r["answer"] == "yes" and r["category"] == "random")),
        "annotation_file": "coco/annotations/instances_val2017.json",
        "annotation_sha256": sha256(ann_path),
        "output_sha256": sha256(out),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {len(rows)} rows for {len(selected)} images -> {out}")
    print("validation: passed")
    print(json.dumps({k: meta[k] for k in ("counts", "positive_size_bins")}, indent=2))


if __name__ == "__main__":
    main()
