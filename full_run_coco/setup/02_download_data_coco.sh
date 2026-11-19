#!/usr/bin/env bash
# COCO val2017 images + instance annotations. Idempotent.
# Run inside WSL:  bash setup/02_download_data_coco.sh
# COCO-only version of 02_download_data.sh.
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

# --- verify ---
python -I - "$DATA" <<'EOF'
import json, os, sys
data = sys.argv[1]
n_img = len([f for f in os.listdir(f"{data}/coco/val2017") if f.endswith(".jpg")])
coco = json.load(open(f"{data}/coco/annotations/instances_val2017.json"))
print(f"COCO val2017: {n_img} images on disk, {len(coco['images'])} in annotations, "
      f"{len(coco['annotations'])} annotations, {len(coco['categories'])} categories")
assert n_img == 5000 and len(coco["images"]) == 5000 and len(coco["categories"]) == 80
EOF

echo "02_download_data_coco DONE"
