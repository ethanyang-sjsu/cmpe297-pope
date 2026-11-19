# View (c): label-quality summary
Stage A: blind single-reviewer verdicts on the predeclared audit sample. Stage B: reviewed ordinary-prompt errors (not blind: reviewers know each item is a model error).

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
