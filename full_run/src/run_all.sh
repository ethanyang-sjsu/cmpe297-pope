#!/usr/bin/env bash
# Phase 3 full runs: every frozen question under every prompt (2 datasets x 3 prompts x
# 9,000 questions = 54,000 generations, ~2.5 h on an RTX 4080). Run inside WSL:
#   bash src/run_all.sh
# Idempotent: a task is skipped only if a samples file with all of its rows already exists,
# so an interrupted run can simply be restarted. Never rerun a finished task selectively
# after looking at results.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cmpe297
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$PROJ/results/raw/full
MODEL=Qwen/Qwen2.5-VL-3B-Instruct
mkdir -p "$OUT"
cd "$PROJ"

# gates: frozen sets unchanged, task files current, parser/prompt tests pass
python src/freeze_eval_set.py --verify
python src/make_task_yamls.py > /dev/null
python -m pytest -q tests/

complete_rows() {  # rows in the newest samples file of a task (0 if none)
  local f
  f=$(ls -t "$OUT"/*/*_samples_"$1".jsonl 2>/dev/null | head -1 || true)
  [ -n "$f" ] && wc -l < "$f" || echo 0
}

for ds in coco gqa; do
  expected=$(wc -l < "data/eval_sets/${ds}_pope.jsonl")
  for prompt in ordinary neutral evidence; do
    task=pope_${ds}_${prompt}
    if [ "$(complete_rows "$task")" -eq "$expected" ]; then
      echo "skip $task (already complete)"
      continue
    fi
    echo "run  $task ($expected questions) -> $OUT/run_$task.log"
    start=$(date -u +%FT%TZ)
    python -m lmms_eval --model qwen2_5_vl --model_args "pretrained=$MODEL" \
      --include_path "$PROJ/tasks/pope_custom" --tasks "$task" \
      --batch_size 1 --log_samples --output_path "$OUT/" > "$OUT/run_$task.log" 2>&1
    rows=$(complete_rows "$task")
    if [ "$rows" -ne "$expected" ]; then
      echo "FAILED $task: samples file has $rows of $expected rows (see run_$task.log)"
      exit 1
    fi
    # one line of run metadata per finished task, for the README and report
    python - "$task" "$start" "$MODEL" >> "$OUT/run_info.jsonl" <<'EOF'
import datetime, importlib.metadata as md, json, os, sys, torch
task, start, model = sys.argv[1:]
snap = os.path.join(os.environ["HF_HOME"], "hub", "models--" + model.replace("/", "--"), "refs", "main")
print(json.dumps({
    "task": task, "start_utc": start,
    "end_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": model, "model_revision": open(snap).read().strip(),
    "lmms_eval": md.version("lmms_eval"), "transformers": md.version("transformers"),
    "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
}))
EOF
    echo "done $task"
  done
done
echo "run_all DONE"
