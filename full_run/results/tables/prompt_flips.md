# Question wording: answer flips between prompts
Distinct questions whose parsed answer differs between two prompts (invalid parses count as abstentions). Full transition counts in prompt_flips.csv.

| Dataset | Prompts | Questions | N | Answer changed |
|---|---|---|---|---|
| COCO | ordinary vs neutral | positives | 1500 | 48 (3.2%) |
| COCO | ordinary vs neutral | negatives | 3630 | 36 (1.0%) |
| COCO | ordinary vs evidence | positives | 1500 | 152 (10.1%) |
| COCO | ordinary vs evidence | negatives | 3630 | 2173 (59.9%) |
| COCO | neutral vs evidence | positives | 1500 | 138 (9.2%) |
| COCO | neutral vs evidence | negatives | 3630 | 2179 (60.0%) |
| GQA | ordinary vs neutral | positives | 1500 | 36 (2.4%) |
| GQA | ordinary vs neutral | negatives | 3662 | 80 (2.2%) |
| GQA | ordinary vs evidence | positives | 1500 | 137 (9.1%) |
| GQA | ordinary vs evidence | negatives | 3662 | 2184 (59.6%) |
| GQA | neutral vs evidence | positives | 1500 | 132 (8.8%) |
| GQA | neutral vs evidence | negatives | 3662 | 2183 (59.6%) |

## Where the ordinary prompt's answers go under the evidence prompt

Does permission to abstain remove hallucinations, or correct answers? Counts of distinct questions.

### COCO

| Ordinary answer | N | Evidence: yes | Evidence: no | Evidence: not sure |
|---|---|---|---|---|
| negatives: ordinary *yes* (hallucination) | 103 | 99 (96.1%) | 3 (2.9%) | 1 (1.0%) |
| negatives: ordinary *no* (correct) | 3527 | 105 (3.0%) | 1358 (38.5%) | 2064 (58.5%) |
| positives: ordinary *yes* (correct) | 1291 | 1289 (99.8%) | 2 (0.2%) | 0 (0.0%) |
| positives: ordinary *no* (miss) | 209 | 89 (42.6%) | 59 (28.2%) | 61 (29.2%) |

### GQA

| Ordinary answer | N | Evidence: yes | Evidence: no | Evidence: not sure |
|---|---|---|---|---|
| negatives: ordinary *yes* (hallucination) | 213 | 202 (94.8%) | 3 (1.4%) | 8 (3.8%) |
| negatives: ordinary *no* (correct) | 3449 | 114 (3.3%) | 1276 (37.0%) | 2059 (59.7%) |
| positives: ordinary *yes* (correct) | 1280 | 1274 (99.5%) | 3 (0.2%) | 3 (0.2%) |
| positives: ordinary *no* (miss) | 220 | 52 (23.6%) | 89 (40.5%) | 79 (35.9%) |
