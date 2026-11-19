# View (b): Stage A audited subset
The 300 audited questions (150 drawn as positives, 150 as negatives), each counted once. *original* scores them with the dataset labels, *corrected* with the reviewer's verdicts; items judged uncertain are excluded from both and counted in "Uncertain". All metrics in %. Per-split rows are in metrics_long.csv.

| Prompt | Labels | N | N pos | N neg | Uncertain | Acc | Prec | Rec | F1 | Spec | Halluc (FPR) | FNR | Yes ratio | Abstain | Abst. pos | Abst. neg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ordinary | original | 288 | 142 | 146 | 12 | 92.4 | 99.2 | 85.2 | 91.7 | 99.3 | 0.7 | 14.8 | 42.4 | 0.0 | 0.0 | 0.0 |
| ordinary | corrected | 288 | 146 | 142 | 12 | 91.0 | 99.2 | 82.9 | 90.3 | 99.3 | 0.7 | 17.1 | 42.4 | 0.0 | 0.0 | 0.0 |
| neutral | original | 288 | 142 | 146 | 12 | 93.1 | 98.4 | 87.3 | 92.5 | 98.6 | 1.4 | 12.7 | 43.8 | 0.0 | 0.0 | 0.0 |
| neutral | corrected | 288 | 146 | 142 | 12 | 92.4 | 99.2 | 85.6 | 91.9 | 99.3 | 0.7 | 14.4 | 43.8 | 0.0 | 0.0 | 0.0 |
| evidence | original | 288 | 142 | 146 | 12 | 65.6 | 95.6 | 90.8 | 93.1 | 41.1 | 4.1 | 2.1 | 46.9 | 31.2 | 7.0 | 54.8 |
| evidence | corrected | 288 | 146 | 142 | 12 | 64.6 | 96.3 | 89.0 | 92.5 | 39.4 | 3.5 | 4.8 | 46.9 | 31.2 | 6.2 | 57.0 |
