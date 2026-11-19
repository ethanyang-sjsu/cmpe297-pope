# Split comparisons: hallucination rate, ordinary prompt
Difference = second split minus first, in percentage points. 95% CI from a paired bootstrap over images (1,000 resamples, seed 297): each resample scores both splits on the same images. Significant = the interval excludes 0.

| Comparison | FPR first (%) | FPR second (%) | Difference (pp) | 95% CI (pp) | Significant |
|---|---|---|---|---|---|
| random vs popular | 0.8 | 3.7 | +2.9 | [+1.9, +3.9] | yes |
| random vs adversarial | 0.8 | 5.7 | +4.9 | [+3.8, +6.1] | yes |
