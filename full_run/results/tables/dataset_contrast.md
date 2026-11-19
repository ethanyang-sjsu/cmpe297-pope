# Dataset contrast: COCO vs GQA
Same prompt and split, full set with original labels. Point estimate [95% bootstrap CI over images] in %, from score.py.

## Accuracy

| Prompt | Split | COCO | GQA | GQA - COCO (pp) |
|---|---|---|---|---|
| ordinary | random | 92.6 [91.7, 93.5] | 91.7 [90.8, 92.7] | -0.9 |
| ordinary | popular | 91.2 [90.1, 92.2] | 88.2 [87.1, 89.3] | -3.0 |
| ordinary | adversarial | 90.2 [89.2, 91.2] | 87.4 [86.3, 88.6] | -2.7 |
| ordinary | all | 91.3 [90.4, 92.3] | 89.1 [88.2, 90.1] | -2.2 |
| neutral | random | 93.4 [92.6, 94.3] | 91.8 [90.9, 92.8] | -1.6 |
| neutral | popular | 92.0 [91.1, 93.0] | 86.9 [85.9, 88.0] | -5.1 |
| neutral | adversarial | 90.7 [89.8, 91.8] | 86.5 [85.4, 87.6] | -4.3 |
| neutral | all | 92.1 [91.2, 93.0] | 88.4 [87.5, 89.4] | -3.7 |
| evidence | random | 64.2 [62.7, 65.8] | 62.5 [61.0, 64.2] | -1.7 |
| evidence | popular | 64.5 [63.0, 65.9] | 60.0 [58.3, 61.6] | -4.5 |
| evidence | adversarial | 66.3 [64.9, 67.9] | 60.0 [58.5, 61.5] | -6.4 |
| evidence | all | 65.0 [63.8, 66.3] | 60.8 [59.5, 62.2] | -4.2 |

## F1

| Prompt | Split | COCO | GQA | GQA - COCO (pp) |
|---|---|---|---|---|
| ordinary | random | 92.1 [91.1, 93.1] | 91.2 [90.1, 92.3] | -0.9 |
| ordinary | popular | 90.7 [89.6, 91.8] | 87.8 [86.7, 89.0] | -2.9 |
| ordinary | adversarial | 89.7 [88.7, 90.9] | 87.2 [86.0, 88.4] | -2.6 |
| ordinary | all | 90.9 [89.8, 91.9] | 88.7 [87.6, 89.8] | -2.2 |
| neutral | random | 93.0 [92.1, 94.0] | 91.3 [90.2, 92.3] | -1.7 |
| neutral | popular | 91.7 [90.7, 92.7] | 86.8 [85.7, 87.9] | -4.9 |
| neutral | adversarial | 90.4 [89.4, 91.5] | 86.3 [85.2, 87.5] | -4.1 |
| neutral | all | 91.7 [90.7, 92.7] | 88.1 [87.1, 89.1] | -3.6 |
| evidence | random | 94.8 [94.0, 95.6] | 92.0 [91.0, 93.0] | -2.8 |
| evidence | popular | 92.7 [91.8, 93.7] | 88.8 [87.6, 89.9] | -3.9 |
| evidence | adversarial | 90.7 [89.7, 91.6] | 86.8 [85.6, 88.0] | -3.8 |
| evidence | all | 92.7 [91.9, 93.5] | 89.2 [88.1, 90.1] | -3.5 |

## Halluc (FPR)

| Prompt | Split | COCO | GQA | GQA - COCO (pp) |
|---|---|---|---|---|
| ordinary | random | 0.8 [0.4, 1.2] | 1.9 [1.2, 2.6] | +1.1 |
| ordinary | popular | 3.7 [2.7, 4.7] | 9.0 [7.6, 10.5] | +5.3 |
| ordinary | adversarial | 5.7 [4.7, 6.8] | 10.5 [9.0, 12.1] | +4.7 |
| ordinary | all | 3.4 [2.8, 4.0] | 7.1 [6.2, 8.0] | +3.7 |
| neutral | random | 0.7 [0.3, 1.1] | 1.9 [1.3, 2.7] | +1.3 |
| neutral | popular | 3.5 [2.6, 4.4] | 11.7 [10.3, 13.3] | +8.3 |
| neutral | adversarial | 6.1 [5.0, 7.2] | 12.7 [10.9, 14.3] | +6.6 |
| neutral | all | 3.4 [2.8, 4.0] | 8.8 [7.8, 9.8] | +5.4 |
| evidence | random | 1.9 [1.2, 2.7] | 3.7 [2.9, 4.7] | +1.8 |
| evidence | popular | 6.3 [5.1, 7.5] | 10.7 [9.1, 12.5] | +4.4 |
| evidence | adversarial | 10.8 [9.3, 12.3] | 15.2 [13.3, 17.2] | +4.4 |
| evidence | all | 6.4 [5.5, 7.3] | 9.9 [8.8, 11.0] | +3.5 |

## FNR

| Prompt | Split | COCO | GQA | GQA - COCO (pp) |
|---|---|---|---|---|
| ordinary | random | 13.9 [12.2, 15.5] | 14.7 [13.0, 16.3] | +0.7 |
| ordinary | popular | 13.9 [12.2, 15.5] | 14.7 [13.0, 16.3] | +0.7 |
| ordinary | adversarial | 13.9 [12.2, 15.5] | 14.7 [13.0, 16.3] | +0.7 |
| ordinary | all | 13.9 [12.2, 15.5] | 14.7 [13.0, 16.3] | +0.7 |
| neutral | random | 12.5 [10.9, 14.1] | 14.4 [12.7, 16.1] | +1.9 |
| neutral | popular | 12.5 [10.9, 14.1] | 14.4 [12.7, 16.1] | +1.9 |
| neutral | adversarial | 12.5 [10.9, 14.1] | 14.4 [12.7, 16.1] | +1.9 |
| neutral | all | 12.5 [10.9, 14.1] | 14.4 [12.7, 16.1] | +1.9 |
| evidence | random | 4.1 [3.1, 5.0] | 6.1 [5.0, 7.3] | +2.1 |
| evidence | popular | 4.1 [3.1, 5.0] | 6.1 [5.0, 7.3] | +2.1 |
| evidence | adversarial | 4.1 [3.1, 5.0] | 6.1 [5.0, 7.3] | +2.1 |
| evidence | all | 4.1 [3.1, 5.0] | 6.1 [5.0, 7.3] | +2.1 |
