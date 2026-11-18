import json

FILE = "/home/ethan/finalproject/logs/Qwen__Qwen2.5-VL-3B-Instruct/20261117_124621_samples_custom_pope.jsonl"


def foo(records):
    num_samples = len(records)
    num_tp = 0
    num_tn = 0
    num_fp = 0
    num_fn = 0
    abs_for_yes = 0
    abs_for_no = 0
    for r in records:
        if r["filtered_resps"] == "yes":
            if r["target"] == "yes":
                num_tp += 1
            if r["target"] == "no":
                num_fp += 1
        elif r["filtered_resps"] == "no":
            if r["target"] == "no":
                num_tn += 1
            if r["target"] == "yes":
                num_fn += 1
        elif r["filtered_resps"] == "unsure":
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

    precision = num_tp / num_yes
    recall = num_tp / (num_tp + num_fn)
    f1 = 2 * precision * recall / (precision + recall)
    specificity = num_tn / (num_tn + num_fp)

    print(f"Accuracy {num_correct/num_samples} ({num_correct}/{num_samples})")
    print(f"Precision {precision} ({num_tp}/{num_yes})")
    print(f"Recall {recall} ({num_tp}/{num_tp+num_fn})")
    print(f"F1 {f1}")
    print(f"Specificity {specificity} ({num_tn}/{num_tn+num_fp})")
    print(f"Yes ratio {num_yes/num_samples} ({num_yes}/{num_samples})")
    print(f"FP rate {num_fp/num_samples} ({num_fp}/{num_samples})")
    print(f"FN rate {num_fn/num_samples} ({num_fn}/{num_samples})")
    print(f"Abstention rate {num_abs/num_samples} ({num_abs}/{num_samples})")
    print(f"Abstain when ground truth is yes {abs_for_yes/num_abs if num_abs else 0} ({abs_for_yes}/{num_abs})")
    print(f"Abstain when ground truth is no {abs_for_no/num_abs if num_abs else 0} ({abs_for_no}/{num_abs})")


with open(FILE, 'r') as f:
    records = [json.loads(line) for line in f if line.strip()]

# This particular split has 50/50/50 in this order
print("RANDOM")
foo(records[:50])
print("\nPOPULAR")
foo(records[50:100])
print("\nADVERSARIAL")
foo(records[100:])
