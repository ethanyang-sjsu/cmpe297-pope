For SJSU Fall 2026 term, CMPE 297-CV final project.

Group 7:
- Inderpreet Buttar
- Andy Wu
- Ethan Yang

# Dependencies
lmms-eval 0.7.4

Make note of the install path for lmms-eval, it will be used later.
```bash
pip show lmms_eval

# sample output
$USER/miniconda3/lib/python3.14/site-packages/lmms_eval
```

# To run
Navigate into the subdirectories and see the corresponding READMEs.

# To get metrics:
1. Edit `manual_metrics.py`
    ```py
    FILE = "<log>"

    # Example:
    FILE = "./logs/Qwen__Qwen2.5-VL-3B-Instruct/20261118_175752_samples_custom_gqa_balanced.jsonl"
    ```
2. Run `manual_metrics.py`
3. If needed, make edits to account for model responses that don't match `["yes", "no", "unsure"]`, and rerun

