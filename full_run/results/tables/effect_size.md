# Object size: positives by size bin
Distinct positive questions; size bin of the largest instance (COCO thresholds: small < 32², medium < 96² pixels). Recall + FNR + abstention = 100%. Rates in %.

## COCO

| Prompt | Size | N | Recall | FNR | Abstain |
|---|---|---|---|---|---|
| ordinary | small | 196 | 56.6 | 43.4 | 0.0 |
| ordinary | medium | 412 | 83.5 | 16.5 | 0.0 |
| ordinary | large | 892 | 93.7 | 6.3 | 0.0 |
| neutral | small | 196 | 62.2 | 37.8 | 0.0 |
| neutral | medium | 412 | 85.7 | 14.3 | 0.0 |
| neutral | large | 892 | 93.9 | 6.1 | 0.0 |
| evidence | small | 196 | 72.4 | 9.7 | 17.9 |
| evidence | medium | 412 | 92.0 | 4.4 | 3.6 |
| evidence | large | 892 | 96.1 | 2.7 | 1.2 |

## GQA

| Prompt | Size | N | Recall | FNR | Abstain |
|---|---|---|---|---|---|
| ordinary | small | 228 | 60.5 | 39.5 | 0.0 |
| ordinary | medium | 467 | 82.2 | 17.8 | 0.0 |
| ordinary | large | 805 | 94.2 | 5.8 | 0.0 |
| neutral | small | 228 | 62.7 | 37.3 | 0.0 |
| neutral | medium | 467 | 82.7 | 17.3 | 0.0 |
| neutral | large | 805 | 93.8 | 6.2 | 0.0 |
| evidence | small | 228 | 68.9 | 9.2 | 21.9 |
| evidence | medium | 467 | 86.1 | 9.9 | 4.1 |
| evidence | large | 805 | 95.3 | 3.1 | 1.6 |
