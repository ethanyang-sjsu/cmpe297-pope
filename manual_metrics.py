import json

# FILE = "/home/ethan/finalproject/logs/Qwen__Qwen2.5-VL-3B-Instruct/20261118_175752_samples_custom_gqa_all.jsonl"
FILE = "/home/ethan/finalproject/logs/Qwen__Qwen2.5-VL-3B-Instruct/20261118_175752_samples_custom_gqa_balanced.jsonl"


def response_preview(records):
    seen = set()
    for r in records:
        seen.add(r["filtered_resps"])
    print(f"Records have these responses: {seen}")

def metric_dec_frac(name, num, denom):
    if denom == 0:
        result = float("nan")
    else:
        result = num / denom
    print(name, f"{result:.4f}", f"{num}/{denom}", sep="\t")
    return result

def foo(records):
    num_samples = len(records)
    num_tp = 0
    num_tn = 0
    num_fp = 0
    num_fn = 0
    abs_for_yes = 0
    abs_for_no = 0
    for r in records:
        if r["filtered_resps"].lower().startswith("yes"):
            if r["target"] == "yes":
                num_tp += 1
            if r["target"] == "no":
                num_fp += 1
        elif r["filtered_resps"].lower().startswith("no"):
            if r["target"] == "no":
                num_tn += 1
            if r["target"] == "yes":
                num_fn += 1
        elif r["filtered_resps"].lower().startswith("unsure"):
            if r["target"] == "yes":
                abs_for_yes += 1
            if r["target"] == "no":
                abs_for_no += 1
        else:
            print("RESPONSE NEEDS MANUAL CORRECTION", r["filtered_resps"])
            return
    num_correct = num_tp + num_tn
    num_yes = num_tp + num_fp
    num_abs = abs_for_yes + abs_for_no

    metric_dec_frac("Accuracy", num_correct, num_samples)
    precision = metric_dec_frac("Precision", num_tp, num_yes)
    recall = metric_dec_frac("Recall", num_tp, num_tp+num_fn+abs_for_yes)
    metric_dec_frac("F1", 2 * precision * recall, precision + recall)
    metric_dec_frac("Specificity", num_tn, num_tn+num_fp+abs_for_no)
    metric_dec_frac("Yes ratio", num_yes, num_samples)
    metric_dec_frac("FP rate", num_fp, num_fp+num_tn+abs_for_no)
    metric_dec_frac("FN rate", num_fn, num_fn+num_tp+abs_for_yes)
    metric_dec_frac("Abstention rate", num_abs, num_samples)


with open(FILE, 'r') as f:
    records = [json.loads(line) for line in f if line.strip()]

print(FILE)
response_preview(records)
print()
foo(records)
