"""Download only the GQA images that the GQA eval set uses (instead of the 20 GB zip).

Usage (inside WSL, conda env cmpe297, from the project folder), after build_pope_gqa.py:
    python src/fetch_gqa_images.py

GQA images are Visual Genome images with the same ids, so each <id>.jpg is fetched from
VG_100K/, falling back to VG_100K_2/. Files that already exist and open as images are
skipped, so the script is safe to rerun (e.g. after the GQA set is rebuilt). Each file is
written under a temporary name and renamed only after it opens as a valid image.

Ids that still fail are written to $CMPE297_DATA/gqa/fetch_failures.txt and the script
exits with status 1. Fallback for those ids: download the 20 GB GQA zip
(https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) and extract only them:
    unzip -n images.zip $(sed 's#^#images/#; s#$#.jpg#' fetch_failures.txt)
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
BASES = ["https://cs.stanford.edu/people/rak248/VG_100K/",
         "https://cs.stanford.edu/people/rak248/VG_100K_2/"]


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def fetch(image: str, out_dir: Path, retries: int = 3) -> str | None:
    """Download one image; return None on success, else a short error message."""
    dest = out_dir / image
    if dest.exists() and is_valid_image(dest):
        return None
    tmp = dest.with_suffix(".part")
    last_error = "not attempted"
    for base in BASES:
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(base + image, timeout=30) as resp, open(tmp, "wb") as fh:
                    fh.write(resp.read())
                if is_valid_image(tmp):
                    tmp.replace(dest)
                    return None
                last_error = f"{base}: not a valid image"
                break  # a bad file will not get better on retry; try the other base
            except urllib.error.HTTPError as e:
                last_error = f"{base}: HTTP {e.code}"
                if e.code == 404:
                    break  # not in this directory; try the other base
            except Exception as e:  # network errors: retry with backoff
                last_error = f"{base}: {type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    tmp.unlink(missing_ok=True)
    return last_error


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.environ.get("CMPE297_DATA", os.path.expanduser("~/cmpe297/data")))
    ap.add_argument("--eval-set", default=str(PROJECT / "data/eval_sets/gqa_pope.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with open(args.eval_set) as fh:
        images = list(dict.fromkeys(json.loads(line)["image"] for line in fh if line.strip()))
    out_dir = Path(args.data_root) / "gqa/images"
    out_dir.mkdir(parents=True, exist_ok=True)
    already = sum((out_dir / im).exists() for im in images)
    print(f"{len(images)} images in {args.eval_set}; {already} already on disk")

    with ThreadPoolExecutor(args.workers) as pool:
        results = dict(zip(images, pool.map(lambda im: fetch(im, out_dir), images)))
    failures = {im: err for im, err in results.items() if err}

    fail_path = out_dir.parent / "fetch_failures.txt"
    if failures:
        fail_path.write_text("".join(im.removesuffix(".jpg") + "\n" for im in failures))
        print(f"{len(failures)} images failed (ids in {fail_path}):")
        for im, err in list(failures.items())[:10]:
            print(f"  {im}: {err}")
        sys.exit(1)
    fail_path.unlink(missing_ok=True)
    size_mb = sum((out_dir / im).stat().st_size for im in images) / 2 ** 20
    print(f"all {len(images)} images present and valid in {out_dir} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
