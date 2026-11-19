# View (b): Stage A audited subset
The 300 audited questions per dataset (150 drawn as positives, 150 as negatives), each counted once. *original* scores them with the dataset labels, *corrected* with the reviewer's verdicts; items judged uncertain are excluded from both and counted in "Uncertain". All metrics in %. Per-split rows are in metrics_long.csv.

## COCO

| Prompt | Labels | N | N pos | N neg | Uncertain | Acc | Prec | Rec | F1 | Spec | Halluc (FPR) | FNR | Yes ratio | Abstain | Abst. pos | Abst. neg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ordinary | original | 288 | 142 | 146 | 12 | 92.4 | 99.2 | 85.2 | 91.7 | 99.3 | 0.7 | 14.8 | 42.4 | 0.0 | 0.0 | 0.0 |
| ordinary | corrected | 288 | 146 | 142 | 12 | 91.0 | 99.2 | 82.9 | 90.3 | 99.3 | 0.7 | 17.1 | 42.4 | 0.0 | 0.0 | 0.0 |
| neutral | original | 288 | 142 | 146 | 12 | 93.1 | 98.4 | 87.3 | 92.5 | 98.6 | 1.4 | 12.7 | 43.8 | 0.0 | 0.0 | 0.0 |
| neutral | corrected | 288 | 146 | 142 | 12 | 92.4 | 99.2 | 85.6 | 91.9 | 99.3 | 0.7 | 14.4 | 43.8 | 0.0 | 0.0 | 0.0 |
| evidence | original | 288 | 142 | 146 | 12 | 65.6 | 95.6 | 90.8 | 93.1 | 41.1 | 4.1 | 2.1 | 46.9 | 31.2 | 7.0 | 54.8 |
| evidence | corrected | 288 | 146 | 142 | 12 | 64.6 | 96.3 | 89.0 | 92.5 | 39.4 | 3.5 | 4.8 | 46.9 | 31.2 | 6.2 | 57.0 |

## GQA

| Prompt | Labels | N | N pos | N neg | Uncertain | Acc | Prec | Rec | F1 | Spec | Halluc (FPR) | FNR | Yes ratio | Abstain | Abst. pos | Abst. neg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ordinary | original | 286 | 141 | 145 | 14 | 88.5 | 90.3 | 85.8 | 88.0 | 91.0 | 9.0 | 14.2 | 46.9 | 0.0 | 0.0 | 0.0 |
| ordinary | corrected | 286 | 152 | 134 | 14 | 90.2 | 96.3 | 84.9 | 90.2 | 96.3 | 3.7 | 15.1 | 46.9 | 0.0 | 0.0 | 0.0 |
| neutral | original | 286 | 141 | 145 | 14 | 87.1 | 87.7 | 85.8 | 86.7 | 88.3 | 11.7 | 14.2 | 48.3 | 0.0 | 0.0 | 0.0 |
| neutral | corrected | 286 | 152 | 134 | 14 | 90.9 | 95.7 | 86.8 | 91.0 | 95.5 | 4.5 | 13.2 | 48.3 | 0.0 | 0.0 | 0.0 |
| evidence | original | 286 | 141 | 145 | 14 | 58.7 | 89.2 | 87.9 | 88.6 | 30.3 | 10.3 | 7.1 | 48.6 | 32.5 | 5.0 | 59.3 |
| evidence | corrected | 286 | 152 | 134 | 14 | 62.2 | 95.0 | 86.8 | 90.7 | 34.3 | 5.2 | 5.3 | 48.6 | 32.5 | 7.9 | 60.4 |
