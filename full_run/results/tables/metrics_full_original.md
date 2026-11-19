# View (a): full set, original labels
All metrics in %. N = 3,000 per split (1,500 yes / 1,500 no, class balance 50%); "all" pools the three splits (N = 9,000). Abstain counts abstentions plus invalid parses (invalid parses are listed in metrics_long.csv).

## COCO

| Prompt | Split | Acc | Prec | Rec | F1 | Spec | Halluc (FPR) | FNR | Yes ratio | Abstain | Abst. pos | Abst. neg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ordinary | random | 92.6 | 99.1 | 86.1 | 92.1 | 99.2 | 0.8 | 13.9 | 43.4 | 0.0 | 0.0 | 0.0 |
| ordinary | popular | 91.2 | 95.9 | 86.1 | 90.7 | 96.3 | 3.7 | 13.9 | 44.9 | 0.0 | 0.0 | 0.0 |
| ordinary | adversarial | 90.2 | 93.8 | 86.1 | 89.7 | 94.3 | 5.7 | 13.9 | 45.9 | 0.0 | 0.0 | 0.0 |
| ordinary | all | 91.3 | 96.2 | 86.1 | 90.9 | 96.6 | 3.4 | 13.9 | 44.7 | 0.0 | 0.0 | 0.0 |
| neutral | random | 93.4 | 99.2 | 87.5 | 93.0 | 99.3 | 0.7 | 12.5 | 44.1 | 0.0 | 0.0 | 0.0 |
| neutral | popular | 92.0 | 96.2 | 87.5 | 91.7 | 96.5 | 3.5 | 12.5 | 45.5 | 0.0 | 0.0 | 0.0 |
| neutral | adversarial | 90.7 | 93.5 | 87.5 | 90.4 | 93.9 | 6.1 | 12.5 | 46.8 | 0.0 | 0.0 | 0.0 |
| neutral | all | 92.1 | 96.3 | 87.5 | 91.7 | 96.6 | 3.4 | 12.5 | 45.5 | 0.0 | 0.0 | 0.0 |
| evidence | random | 64.2 | 97.9 | 91.9 | 94.8 | 36.5 | 1.9 | 4.1 | 46.9 | 32.8 | 4.1 | 61.5 |
| evidence | popular | 64.5 | 93.6 | 91.9 | 92.7 | 37.1 | 6.3 | 4.1 | 49.1 | 30.3 | 4.1 | 56.6 |
| evidence | adversarial | 66.3 | 89.5 | 91.9 | 90.7 | 40.8 | 10.8 | 4.1 | 51.3 | 26.2 | 4.1 | 48.4 |
| evidence | all | 65.0 | 93.5 | 91.9 | 92.7 | 38.1 | 6.4 | 4.1 | 49.1 | 29.8 | 4.1 | 55.5 |

## GQA

| Prompt | Split | Acc | Prec | Rec | F1 | Spec | Halluc (FPR) | FNR | Yes ratio | Abstain | Abst. pos | Abst. neg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ordinary | random | 91.7 | 97.9 | 85.3 | 91.2 | 98.1 | 1.9 | 14.7 | 43.6 | 0.0 | 0.0 | 0.0 |
| ordinary | popular | 88.2 | 90.5 | 85.3 | 87.8 | 91.0 | 9.0 | 14.7 | 47.2 | 0.0 | 0.0 | 0.0 |
| ordinary | adversarial | 87.4 | 89.1 | 85.3 | 87.2 | 89.5 | 10.5 | 14.7 | 47.9 | 0.0 | 0.0 | 0.0 |
| ordinary | all | 89.1 | 92.3 | 85.3 | 88.7 | 92.9 | 7.1 | 14.7 | 46.2 | 0.0 | 0.0 | 0.0 |
| neutral | random | 91.8 | 97.8 | 85.6 | 91.3 | 98.1 | 1.9 | 14.4 | 43.8 | 0.0 | 0.0 | 0.0 |
| neutral | popular | 86.9 | 87.9 | 85.6 | 86.8 | 88.3 | 11.7 | 14.4 | 48.7 | 0.0 | 0.0 | 0.0 |
| neutral | adversarial | 86.5 | 87.1 | 85.6 | 86.3 | 87.3 | 12.7 | 14.4 | 49.1 | 0.0 | 0.0 | 0.0 |
| neutral | all | 88.4 | 90.7 | 85.6 | 88.1 | 91.2 | 8.8 | 14.4 | 47.2 | 0.0 | 0.0 | 0.0 |
| evidence | random | 62.5 | 95.9 | 88.4 | 92.0 | 36.7 | 3.7 | 6.1 | 46.1 | 32.5 | 5.5 | 59.6 |
| evidence | popular | 60.0 | 89.2 | 88.4 | 88.8 | 31.6 | 10.7 | 6.1 | 49.6 | 31.6 | 5.5 | 57.7 |
| evidence | adversarial | 60.0 | 85.3 | 88.4 | 86.8 | 31.5 | 15.2 | 6.1 | 51.8 | 29.4 | 5.5 | 53.3 |
| evidence | all | 60.8 | 89.9 | 88.4 | 89.2 | 33.3 | 9.9 | 6.1 | 49.1 | 31.2 | 5.5 | 56.8 |
