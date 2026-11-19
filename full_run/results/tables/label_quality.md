# View (c): label-quality summary
Stage A: blind single-reviewer verdicts on the predeclared audit sample. Stage B: reviewed ordinary-prompt errors (not blind: reviewers know each item is a model error).

## COCO

| Items | Judged present | Judged absent | Judged uncertain |
|---|---|---|---|
| Stage A negatives (dataset says *no*), n=150 | 6 (4.0%) | 140 (93.3%) | 4 (2.7%) |
| Stage A positives (dataset says *yes*), n=150 | 140 (93.3%) | 2 (1.3%) | 8 (5.3%) |

| Stage B list | Reviewed | Tagged *label error* |
|---|---|---|
| Ordinary-prompt false positives | 100 | 20 (20.0%) |
| Ordinary-prompt false negatives | 50 | 15 (30.0%) |

| Stage B false-positive tag | Count |
|---|---|
| similar object | 52 |
| co-occurrence prior | 23 |
| label error | 20 |
| other / no visible cause | 5 |

| Stage B false-negative tag | Count |
|---|---|
| small | 15 |
| label error | 15 |
| occluded or truncated | 14 |
| unusual appearance | 6 |

## GQA

| Items | Judged present | Judged absent | Judged uncertain |
|---|---|---|---|
| Stage A negatives (dataset says *no*), n=150 | 17 (11.3%) | 128 (85.3%) | 5 (3.3%) |
| Stage A positives (dataset says *yes*), n=150 | 135 (90.0%) | 6 (4.0%) | 9 (6.0%) |

| Stage B list | Reviewed | Tagged *label error* |
|---|---|---|
| Ordinary-prompt false positives | 100 | 79 (79.0%) |
| Ordinary-prompt false negatives | 50 | 21 (42.0%) |

| Stage B false-positive tag | Count |
|---|---|
| label error | 79 |
| similar object | 9 |
| other / no visible cause | 6 |
| co-occurrence prior | 5 |
| ambiguous category | 1 |

| Stage B false-negative tag | Count |
|---|---|
| label error | 21 |
| occluded or truncated | 14 |
| small | 10 |
| unusual appearance | 5 |

**Note:** 11.3% of audited GQA negatives were judged present (above 5.0%), so part of the measured hallucination rate is label error; see view (b).
