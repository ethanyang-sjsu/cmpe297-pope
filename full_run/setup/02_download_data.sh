#!/usr/bin/env bash
# Phase 2a: COCO val2017 images + instance annotations, GQA val scene graphs. Idempotent.
# Run inside WSL:  bash setup/02_download_data.sh
# GQA images are NOT downloaded here: src/fetch_gqa_images.py fetches only the ~500 images
# the GQA eval set uses, after src/build_pope_gqa.py has picked them.
set -euxo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cmpe297
DATA=${CMPE297_DATA:-$HOME/cmpe297/data}

# --- COCO val2017 ---
mkdir -p "$DATA/coco" && cd "$DATA/coco"
wget -c -nv http://images.cocodataset.org/zips/val2017.zip                            # 816 MB
wget -c -nv http://images.cocodataset.org/annotations/annotations_trainval2017.zip    # 253 MB
unzip -q -n val2017.zip
unzip -q -n annotations_trainval2017.zip 'annotations/instances_val2017.json'

# --- GQA val scene graphs (the train graphs in the same zip are not needed) ---
mkdir -p "$DATA/gqa" && cd "$DATA/gqa"
wget -c -nv https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip              # 45 MB
unzip -q -n sceneGraphs.zip val_sceneGraphs.json -d sceneGraphs

# --- verify ---
python -I - "$DATA" <<'EOF'
import json, os, sys
data = sys.argv[1]
n_img = len([f for f in os.listdir(f"{data}/coco/val2017") if f.endswith(".jpg")])
coco = json.load(open(f"{data}/coco/annotations/instances_val2017.json"))
sg = json.load(open(f"{data}/gqa/sceneGraphs/val_sceneGraphs.json"))
numeric = sum(k.isdigit() for k in sg)
print(f"COCO val2017: {n_img} images on disk, {len(coco['images'])} in annotations, "
      f"{len(coco['annotations'])} annotations, {len(coco['categories'])} categories")
print(f"GQA val scene graphs: {len(sg)} images ({numeric} with numeric Visual Genome ids)")
assert n_img == 5000 and len(coco["images"]) == 5000 and len(coco["categories"]) == 80
assert len(sg) > 10000
EOF

echo "02_download_data DONE"
