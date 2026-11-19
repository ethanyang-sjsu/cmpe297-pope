"""Build the GQA POPE evaluation set (9,000 questions by default) from val scene graphs.

Usage (inside WSL, conda env cmpe297, from the project folder):
    python src/build_pope_gqa.py
    python src/fetch_gqa_images.py        # then download the selected images

Writes data/eval_sets/gqa_pope.jsonl plus gqa_pope.meta.json. The output is validated
before it is written; nothing is written on failure. Image files are not checked here
(they are fetched afterwards); fetch_gqa_images.py checks them.

GQA-specific choices (see src/gqa_vocab.py for the vocabulary itself):
- An object is present if any of its synonyms is an object name in the scene graph.
  An image's objects are ordered by first appearance in the scene graph; the first 3
  are the positives.
- An absent object may be asked about only if none of its synonyms or blockers appears.
- An image is eligible if it has at least 3 distinct vocabulary objects. 500 eligible
  images are sampled with the seed (POPE also samples its images at random).
- Object size uses the largest matching bounding box, as for COCO.
- Frequency and co-occurrence statistics come from the selected images (as in POPE).
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from build_pope_coco import sha256, size_bin
from gqa_vocab import SYNONYMS, name_index, scene_objects
from pope_sampling import SPLITS, ObjectStats, build_rows, validate_rows

PROJECT = Path(__file__).resolve().parents[1]
K = 3


def load_gqa(sg_path: str):
    """Per image: present objects (ordered), safe negatives, and object sizes."""
    means, blocks = name_index()
    vocab = sorted(SYNONYMS)
    present, negative_pool, size_info = {}, {}, {}
    for image_id, g in json.load(open(sg_path)).items():
        f = f"{image_id}.jpg"
        objs = scene_objects(g)
        names = {o["name"] for o in objs}
        ordered = list(dict.fromkeys(means[o["name"]] for o in objs if o["name"] in means))
        present[f] = ordered
        negative_pool[f] = {v for v in vocab if v not in ordered and not names & blocks[v]}
        image_area = g["width"] * g["height"]
        for obj in ordered:
            area = max(o["w"] * o["h"] for o in objs if means.get(o["name"]) == obj)
            size_info[f, obj] = (area / image_area, size_bin(area))
    return vocab, present, negative_pool, size_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.environ.get("CMPE297_DATA", os.path.expanduser("~/cmpe297/data")))
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=297)
    ap.add_argument("--out", default=str(PROJECT / "data/eval_sets/gqa_pope.jsonl"))
    args = ap.parse_args()

    sg_path = os.path.join(args.data_root, "gqa/sceneGraphs/val_sceneGraphs.json")
    vocab, present, negative_pool, size_info = load_gqa(sg_path)
    eligible = sorted(f for f, objs in present.items() if len(objs) >= K)
    selected = random.Random(args.seed).sample(eligible, args.n_images)

    stats = ObjectStats({f: present[f] for f in selected}, vocab)
    rows = build_rows("gqa", selected, {f: f.removesuffix(".jpg") for f in selected}, present,
                      {f: negative_pool[f] for f in selected}, size_info, stats, K, args.seed)

    errors = validate_rows(rows, len(selected), K, present, negative_pool, data_root=None)
    if errors:
        print("VALIDATION FAILED:", *errors[:20], sep="\n  ")
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    counts = Counter((r["category"], r["answer"]) for r in rows)
    pool_sizes = sorted(len(negative_pool[f]) for f in selected)
    meta = {
        "dataset": "gqa",
        "output": out.name,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": "src/build_pope_gqa.py",
        "seed": args.seed,
        "k_per_image": K,
        "n_images": len(selected),
        "n_eligible_images": len(eligible),
        "n_rows": len(rows),
        "n_vocab_objects": len(vocab),
        "counts": {f"{s}/{a}": counts[s, a] for s in SPLITS for a in ("yes", "no")},
        "positive_size_bins": dict(Counter(r["size_bin"] for r in rows
                                           if r["answer"] == "yes" and r["category"] == "random")),
        "negative_pool_size": {"min": pool_sizes[0], "median": pool_sizes[len(pool_sizes) // 2],
                               "max": pool_sizes[-1]},
        "scene_graph_file": "gqa/sceneGraphs/val_sceneGraphs.json",
        "scene_graph_sha256": sha256(sg_path),
        "vocab_file": "src/gqa_vocab.py",
        "vocab_sha256": sha256(PROJECT / "src/gqa_vocab.py"),
        "output_sha256": sha256(out),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {len(rows)} rows for {len(selected)} images -> {out}")
    print("validation: passed (image files not checked; run src/fetch_gqa_images.py)")
    print(json.dumps({k: meta[k] for k in ("n_eligible_images", "counts", "positive_size_bins",
                                           "negative_pool_size")}, indent=2))


if __name__ == "__main__":
    main()
