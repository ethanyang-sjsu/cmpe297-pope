"""POPE-style question sampling shared by the COCO and GQA builders.

Procedure (Li et al., 2023, "Evaluating Object Hallucination in Large Vision-Language
Models", https://github.com/AoiDragon/POPE): for every image, K objects that are present
become "yes" questions, and each one is paired with one absent object as a "no" question,
chosen per negative-sampling split as in POPE's utils.py:

    random       drawn uniformly from the absent objects
    popular      the most frequent absent object not already used for this image
                 (so the image gets the K most frequent absent objects)
    adversarial  the absent object that co-occurs most often with *this* positive
                 (number of images containing both), not already used for this image;
                 if none co-occurs at all, a random absent object (POPE's fallback)

The "yes" questions are identical across the three splits, so each split has
n_images * K yes and n_images * K no questions. As in POPE, frequency and co-occurrence
are counted over the selected images only.

Positives are the first K distinct objects of the image in source order (annotation-file
order for COCO, scene-graph order for GQA), as POPE takes the first K entries of its
object list. Ranking ties are broken by first appearance while scanning the selected
images in order, which is what POPE's insertion-ordered dicts plus a stable sort do.

For analysis only, every "no" row also stores a normalized co-occurrence score of the
probed object o for an image whose present objects are G:
    cooc_score(o) = mean over g in G of P(o in image | g in image)
Unlike raw counts it is not dominated by how common o is overall, so the analysis can
bin hallucination rate by it.

Dataset-specific choices (what counts as "present", which absent objects are safe to ask
about) are made by the builders, which pass in plain dicts. `present` maps each image to
its distinct objects as a list in source order. Randomness (random negatives and the
adversarial fallback) is seeded per image (seed + image name), so it does not depend on
the order images are processed in.
"""
import os
import random
from collections import Counter
from itertools import combinations

SPLITS = ("random", "popular", "adversarial")
SPLIT_TAG = {"random": "ran", "popular": "pop", "adversarial": "adv"}
# image directory of each dataset, relative to $CMPE297_DATA
IMAGE_DIRS = {"coco": "coco/val2017", "gqa": "gqa/images"}


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def question_text(obj: str) -> str:
    return f"Is there {article(obj)} {obj} in the image?"


def image_path(data_root: str, dataset: str, image: str) -> str:
    return os.path.join(data_root, IMAGE_DIRS[dataset], image)


class ObjectStats:
    """Object frequency and pairwise co-occurrence over per-image object sets.

    image_objects is scanned in its (insertion) order; `seen` and `pair_seen` record when an
    object, or an object as a neighbour of another, first appeared, for POPE's tie-breaking.
    vocab: every askable object, so objects absent from all images still get a rank
    (after all seen objects, alphabetically).
    """

    def __init__(self, image_objects: dict[str, list[str]], vocab: list[str]):
        self.n_images = len(image_objects)
        self.freq = Counter()
        self.pair = Counter()
        self.seen = {}
        self.pair_seen = {}
        for objs in image_objects.values():
            for o in objs:
                self.freq[o] += 1
                self.seen.setdefault(o, len(self.seen))
                for other in objs:
                    if other != o:
                        self.pair_seen.setdefault((o, other), len(self.pair_seen))
            for a, b in combinations(sorted(objs), 2):
                self.pair[a, b] += 1
        unseen = sorted(set(vocab) - set(self.seen))
        self.by_freq = sorted(self.seen, key=lambda o: (-self.freq[o], self.seen[o])) + unseen
        self.freq_rank = {o: i + 1 for i, o in enumerate(self.by_freq)}

    def cooc(self, a: str, b: str) -> int:
        return self.pair[min(a, b), max(a, b)]

    def neighbours(self, obj: str) -> list[str]:
        """Objects that co-occur with obj, most frequent first, ties by first appearance."""
        others = [o for (a, o) in self.pair_seen if a == obj]
        return sorted(others, key=lambda o: (-self.cooc(obj, o), self.pair_seen[obj, o]))

    def cooc_score(self, obj: str, present: list[str]) -> float:
        return sum(self.cooc(obj, g) / self.freq[g] for g in present) / len(present)


def sample_image(image: str, present: list[str], negative_pool: set[str],
                 stats: ObjectStats, k: int, seed: int):
    """Return (positives, {split: negatives}); negatives[split][i] pairs with positives[i].

    present        distinct objects in the image, in source order; positives are the first k
    negative_pool  objects that may be asked about as absent; the builder has already
                   removed everything present, and for GQA also near-synonyms
    """
    rng = random.Random(f"{seed}:{image}")
    positives = present[:k]
    pool = sorted(negative_pool)
    adversarial = []
    for pos in positives:
        ranked = [o for o in stats.neighbours(pos) if o in negative_pool and o not in adversarial]
        adversarial.append(ranked[0] if ranked else
                           rng.choice([o for o in pool if o not in adversarial]))
    negatives = {
        "random": rng.sample(pool, k),
        "popular": [o for o in stats.by_freq if o in negative_pool][:k],
        "adversarial": adversarial,
    }
    return positives, negatives


def build_rows(dataset: str, images: list[str], image_ids: dict[str, object],
               present: dict[str, list[str]], negative_pool: dict[str, set[str]],
               size_info: dict[tuple[str, str], tuple[float, str]],
               stats: ObjectStats, k: int, seed: int) -> list[dict]:
    """Build all rows, split by split; within an image, yes and no questions alternate.

    size_info maps (image, object) -> (obj_area_frac, size_bin) for present objects.
    """
    sampled = {im: sample_image(im, present[im], negative_pool[im], stats, k, seed)
               for im in images}
    rows = []
    for split in SPLITS:
        n = 0
        for im in images:
            positives, negatives = sampled[im]
            for pos, neg in zip(positives, negatives[split]):
                for obj, answer in ((pos, "yes"), (neg, "no")):
                    n += 1
                    area_frac, size_bin = size_info[im, obj] if answer == "yes" else (None, None)
                    rows.append({
                        "question_id": f"{dataset}_{SPLIT_TAG[split]}_{n:05d}",
                        "dataset": dataset,
                        "image": im,
                        "image_id": image_ids[im],
                        "object": obj,
                        "answer": answer,
                        "category": split,
                        "question": question_text(obj),
                        "obj_area_frac": None if area_frac is None else round(area_frac, 6),
                        "size_bin": size_bin,
                        "cooc_score": (round(stats.cooc_score(obj, present[im]), 6)
                                       if answer == "no" else None),
                        "freq_rank": stats.freq_rank.get(obj),
                    })
    return rows


def validate_rows(rows: list[dict], n_images: int, k: int, present: dict[str, list[str]],
                  negative_pool: dict[str, set[str]], data_root: str | None = None) -> list[str]:
    """POPE structural and label checks. Returns a list of problems (empty = valid).

    data_root: if given, also check that every image file exists on disk.
    """
    errors = []
    per_split = n_images * k
    if len(rows) != 2 * per_split * len(SPLITS):
        errors.append(f"expected {2 * per_split * len(SPLITS)} rows, got {len(rows)}")
    ids = [r["question_id"] for r in rows]
    if len(set(ids)) != len(ids):
        errors.append(f"{len(ids) - len(set(ids))} duplicate question_ids")
    for split in SPLITS:
        for answer in ("yes", "no"):
            n = sum(r["category"] == split and r["answer"] == answer for r in rows)
            if n != per_split:
                errors.append(f"{split}/{answer}: expected {per_split}, got {n}")

    by_image_split = {}
    for r in rows:
        key = r["image"], r["category"]
        by_image_split.setdefault(key, []).append(r)
        if r["answer"] == "yes" and r["object"] not in present[r["image"]]:
            errors.append(f"{r['question_id']}: yes-object {r['object']!r} not present in image")
        if r["answer"] == "no" and r["object"] not in negative_pool[r["image"]]:
            errors.append(f"{r['question_id']}: no-object {r['object']!r} not a valid absent object")
    images = {im for im, _ in by_image_split}
    if len(images) != n_images:
        errors.append(f"expected {n_images} images, got {len(images)}")
    for im in images:
        pos_sets = []
        for split in SPLITS:
            group = by_image_split.get((im, split), [])
            objs = [r["object"] for r in group]
            if len(group) != 2 * k or len(set(objs)) != len(objs):
                errors.append(f"{im}/{split}: expected {2 * k} distinct objects, got {objs}")
            pos_sets.append({r["object"] for r in group if r["answer"] == "yes"})
        if any(s != pos_sets[0] for s in pos_sets):
            errors.append(f"{im}: positives differ across splits")
    if data_root is not None:
        missing = [im for im in images
                   if not os.path.exists(image_path(data_root, rows[0]["dataset"], im))]
        if missing:
            errors.append(f"{len(missing)} images missing on disk, e.g. {missing[:3]}")
    return errors
