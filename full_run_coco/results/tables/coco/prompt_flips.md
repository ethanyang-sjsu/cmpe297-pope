# Question wording: answer flips between prompts
Distinct questions whose parsed answer differs between two prompts (invalid parses count as abstentions). Full transition counts in prompt_flips.csv.

| Prompts | Questions | N | Answer changed |
|---|---|---|---|
| ordinary vs neutral | positives | 1500 | 48 (3.2%) |
| ordinary vs neutral | negatives | 3630 | 36 (1.0%) |
| ordinary vs evidence | positives | 1500 | 152 (10.1%) |
| ordinary vs evidence | negatives | 3630 | 2173 (59.9%) |
| neutral vs evidence | positives | 1500 | 138 (9.2%) |
| neutral vs evidence | negatives | 3630 | 2179 (60.0%) |

## Where the ordinary prompt's answers go under the evidence prompt

Does permission to abstain remove hallucinations, or correct answers? Counts of distinct questions.

| Ordinary answer | N | Evidence: yes | Evidence: no | Evidence: not sure |
|---|---|---|---|---|
| negatives: ordinary *yes* (hallucination) | 103 | 99 (96.1%) | 3 (2.9%) | 1 (1.0%) |
| negatives: ordinary *no* (correct) | 3527 | 105 (3.0%) | 1358 (38.5%) | 2064 (58.5%) |
| positives: ordinary *yes* (correct) | 1291 | 1289 (99.8%) | 2 (0.2%) | 0 (0.0%) |
| positives: ordinary *no* (miss) | 209 | 89 (42.6%) | 59 (28.2%) | 61 (29.2%) |
