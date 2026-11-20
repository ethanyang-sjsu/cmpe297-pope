# To setup GQA POPE:
1. Move `./custom_tasks/` into `<lmms_eval path>/tasks/`
2. Edit `pope_gqa_*.yaml` as needed

    ```yaml
    lmms_eval_specific_kwargs:
      default:
        pre_prompt: <your prompt>
    ```
3. Copy `../pope/utils.py` into your `custom_tasks` folder
4. Run
    ```bash
    export HF_TOKEN=<your token>
    python3 -m lmms_eval \
        --model qwen2_5_vl \
        --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
        --include_path ./custom_tasks \
        --tasks pope_gqa \
        --batch_size 1 \
        --log_samples \
        --output_path ./logs/
    ```

    Alternatively, to run individual splits, change `--tasks`

    ```yaml
    # Remove list items as needed
    --tasks pope_gqa_adversarial,pope_gqa_random,pope_gqa_popular
    ```
