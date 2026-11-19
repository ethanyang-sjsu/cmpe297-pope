# Split comparisons: hallucination rate, ordinary prompt
Difference = second split minus first, in percentage points. 95% CI from a paired bootstrap over images (1,000 resamples, seed 297): each resample scores both splits on the same images. Significant = the interval excludes 0.

| Dataset | Comparison | FPR first (%) | FPR second (%) | Difference (pp) | 95% CI (pp) | Significant |
|---|---|---|---|---|---|---|
| COCO | random vs popular | 0.8 | 3.7 | +2.9 | [+1.9, +3.9] | yes |
| COCO | random vs adversarial | 0.8 | 5.7 | +4.9 | [+3.8, +6.1] | yes |
| GQA | random vs popular | 1.9 | 9.0 | +7.1 | [+5.6, +8.8] | yes |
| GQA | random vs adversarial | 1.9 | 10.5 | +8.6 | [+6.9, +10.3] | yes |
