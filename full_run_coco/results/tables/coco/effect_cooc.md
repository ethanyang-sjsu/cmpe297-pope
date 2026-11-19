# Co-occurrence and object frequency: negatives
Hallucination rate (%) of negatives. *cooc_score* = mean over the image's annotated objects of P(probe object | that object); quartile edges from the pooled negatives. *freq_rank* = the probe object's frequency rank in the selected images (1 = most frequent). Pooled = each distinct question once. N per cell in effect_cooc.csv.

## By co-occurrence quartile

| Split | Prompt | Q1 (0.00-0.02) | Q2 (0.02-0.09) | Q3 (0.09-0.21) | Q4 (0.21-0.86) |
|---|---|---|---|---|---|
| pooled | ordinary | 0.3 (n=907) | 1.1 (n=908) | 3.0 (n=907) | 6.9 (n=908) |
| pooled | neutral | 0.1 (n=907) | 1.0 (n=908) | 2.6 (n=907) | 7.6 (n=908) |
| pooled | evidence | 0.7 (n=907) | 2.5 (n=908) | 7.1 (n=907) | 12.2 (n=908) |
| random | ordinary | 0.3 (n=876) | 1.1 (n=452) | 1.6 (n=125) | 4.3 (n=47) |
| random | neutral | 0.1 (n=876) | 0.9 (n=452) | 2.4 (n=125) | 4.3 (n=47) |
| random | evidence | 0.7 (n=876) | 2.7 (n=452) | 5.6 (n=125) | 8.5 (n=47) |
| popular | ordinary | 0.0 (n=31) | 0.7 (n=402) | 1.5 (n=459) | 7.4 (n=608) |
| popular | neutral | 0.0 (n=31) | 0.7 (n=402) | 0.7 (n=459) | 7.6 (n=608) |
| popular | evidence | 0.0 (n=31) | 1.7 (n=402) | 3.9 (n=459) | 11.5 (n=608) |
| adversarial | ordinary | 0.0 (n=1) | 2.2 (n=136) | 3.9 (n=584) | 7.7 (n=779) |
| adversarial | neutral | 0.0 (n=1) | 2.2 (n=136) | 3.8 (n=584) | 8.5 (n=779) |
| adversarial | evidence | 0.0 (n=1) | 3.7 (n=136) | 8.7 (n=584) | 13.6 (n=779) |

## By frequency rank

| Split | Prompt | 1-5 | 6-20 | 21-50 | 51+ |
|---|---|---|---|---|---|
| pooled | ordinary | 3.5 (n=1496) | 4.2 (n=787) | 1.6 (n=742) | 0.8 (n=605) |
| pooled | neutral | 3.4 (n=1496) | 4.4 (n=787) | 1.9 (n=742) | 0.5 (n=605) |
| pooled | evidence | 6.1 (n=1496) | 10.2 (n=787) | 3.4 (n=742) | 1.3 (n=605) |
| random | ordinary | 1.6 (n=62) | 1.1 (n=285) | 0.9 (n=573) | 0.5 (n=580) |
| random | neutral | 1.6 (n=62) | 1.1 (n=285) | 0.9 (n=573) | 0.2 (n=580) |
| random | evidence | 4.8 (n=62) | 3.2 (n=285) | 1.9 (n=573) | 1.0 (n=580) |
| popular | ordinary | 3.6 (n=1430) | 4.3 (n=70) | n/a (n=0) | n/a (n=0) |
| popular | neutral | 3.4 (n=1430) | 4.3 (n=70) | n/a (n=0) | n/a (n=0) |
| popular | evidence | 5.8 (n=1430) | 17.1 (n=70) | n/a (n=0) | n/a (n=0) |
| adversarial | ordinary | 5.8 (n=809) | 5.9 (n=491) | 4.6 (n=174) | 7.7 (n=26) |
| adversarial | neutral | 5.8 (n=809) | 6.5 (n=491) | 5.7 (n=174) | 7.7 (n=26) |
| adversarial | evidence | 9.5 (n=809) | 13.8 (n=491) | 8.6 (n=174) | 7.7 (n=26) |
