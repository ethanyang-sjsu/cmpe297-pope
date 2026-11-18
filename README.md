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

## To setup GQA:

1. Get GQA question data
    - All split
        - `val-00000-of-00002.parquet` from https://huggingface.co/datasets/lmms-lab-encoder/GQA/tree/main/val_all_instructions
        - Run `gqa_all_to_jsonl.py`
    - Balanced split
        - `val-00000-of-00001.parquet` from https://huggingface.co/datasets/lmms-lab-encoder/GQA/tree/main/val_balanced_instructions
        - Run `gqa_balanced_to_jsonl.py`
2. Move this repo's `./custom_tasks/` into `<lmms_eval path>/tasks/`
3. Edit `./custom_tasks/custom_gqa_*.yaml`
    ```yaml
    dataset_kwargs:
        data_files:
            test: "<jsonl file from previous step>"
    ```
4. Run
    ```bash
    export HF_TOKEN=<your token>
    python3 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --include_path ./custom_tasks \
    --tasks custom_gqa_all,custom_gqa_balanced \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/
    ```
