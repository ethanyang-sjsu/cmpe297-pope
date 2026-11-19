#!/usr/bin/env bash
# Phase 1 gate (GO/NO-GO): GPU visible, lmms-eval importable, Qwen2.5-VL-3B loads in bf16
# and answers one positive and one negative object-existence question. Run inside WSL:
#   bash setup/03_gate_check.sh
# First run downloads ~7.5 GB of model weights into $HF_HOME.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cmpe297

echo "== nvidia-smi"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== lmms-eval CLI"
python -m lmms_eval --help > /dev/null && echo "lmms_eval --help OK"

# one COCO val2017 image (two cats on a couch); reused from the dataset if already downloaded
GATE_DIR=$HOME/cmpe297/gate
IMG=${CMPE297_DATA:-$HOME/cmpe297/data}/coco/val2017/000000039769.jpg
if [ ! -f "$IMG" ]; then
  mkdir -p "$GATE_DIR"
  IMG=$GATE_DIR/000000039769.jpg
  [ -f "$IMG" ] || wget -nv -O "$IMG" http://images.cocodataset.org/val2017/000000039769.jpg
fi

echo "== model load + generation"
GATE_IMAGE="$IMG" python - <<'EOF'
import os, time, torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0))

t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
processor = AutoProcessor.from_pretrained(MODEL)
print(f"loaded in {time.time() - t0:.1f}s")

image = Image.open(os.environ["GATE_IMAGE"]).convert("RGB")
cases = [("cat", "yes"), ("giraffe", "no")]
answers = []
for obj, expected in cases:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"Is there a {obj} in the image? Answer with yes or no."}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(f"Q: {obj!r:10} expected={expected:3}  answer={text.strip()!r}  ({time.time() - t0:.2f}s)")
    answers.append(text.strip().lower().startswith(expected))

print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
assert all(answers), "model answers did not match expected yes/no"
print("GATE PASSED")
EOF
