# Feasibility report

reference = **A0**, conf_thr = 0.25, bootstrap = 200 (95% CI of paired difference vs reference)

| model | deploy mAP50-95 | deploy mAP50 | recall@thr | recall person | recall bicycle | recall car | recall motorcycle | recall bus | recall truck | recall small person | deploy FP/img | bg FP/img | bg imgs w/ FP | COCO mAP50-95 | feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A0** | 0.614 | 0.958 | 96.8 | 96.7 | – | 96.9 | – | – | – | 91.7 | 0.125 | 0.133 | 13.3 | 0.712 | ref |
| A1 | 0.614 | 0.958 | 96.8 | 96.7 | – | 96.9 | – | – | – | 91.7 | 0.125 | 0.133 | 13.3 | 0.712 | ❌ |

## Deltas vs reference (95% CI)

| model | bg FP/img Δ | deploy recall Δ | deploy FP/img Δ | criteria |
|---|---|---|---|---|
| A1 | +0.000 [+0.000, +0.000] | +0.0 [+0.0, +0.0] | +0.000 [+0.000, +0.000] | bg_fp_reduction=FAIL, recall_delta=pass, coco_map_delta=pass |

Criteria: bg FP reduction ≥ 30%, recall drop ≤ 2.0 pts, COCO mAP50-95 drop ≤ 1.0 pts. A delta whose CI includes 0 is not distinguishable from noise.
