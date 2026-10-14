# Feasibility report

reference = **A0**, conf_thr = 0.25, bootstrap = 200 (95% CI of paired difference vs reference)

| model | deploy mAP50-95 | deploy mAP50 | recall@thr | recall person | recall bicycle | recall car | recall motorcycle | recall bus | recall truck | recall small person | deploy FP/img | bg FP/img | bg imgs w/ FP | COCO mAP50-95 | feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prod | 0.021 | 0.033 | 0.0 | 0.0 | – | 0.0 | – | – | – | 0.0 | 0.000 | 0.000 | 0.0 | 0.000 | ❌ |
| **A0** | 0.288 | 0.329 | 0.0 | 0.0 | – | 0.0 | – | – | – | 0.0 | 0.025 | 0.133 | 13.3 | 0.047 | ref |
| A1 | 0.621 | 0.717 | 6.5 | 6.7 | – | 6.2 | – | – | – | 16.7 | 0.050 | 0.033 | 3.3 | 0.084 | ✅ |
| A2 | 0.474 | 0.587 | 6.5 | 3.3 | – | 9.4 | – | – | – | 8.3 | 0.200 | 0.467 | 46.7 | 0.075 | ❌ |
| A3 | 0.541 | 0.639 | 22.6 | 6.7 | – | 37.5 | – | – | – | 16.7 | 0.150 | 0.500 | 50.0 | 0.069 | ❌ |
| A4 | 0.686 | 0.817 | 29.0 | 23.3 | – | 34.4 | – | – | – | 33.3 | 0.100 | 0.267 | 26.7 | 0.123 | ❌ |
| A5 | 0.336 | 0.380 | 0.0 | 0.0 | – | 0.0 | – | – | – | 0.0 | 0.000 | 0.000 | 0.0 | 0.039 | ✅ |

## Deltas vs reference (95% CI)

| model | bg FP/img Δ | deploy recall Δ | deploy FP/img Δ | criteria |
|---|---|---|---|---|
| prod | -0.133 [-0.234, -0.033] | +0.0 [+0.0, +0.0] | -0.025 [-0.075, +0.000] | bg_fp_reduction=pass, recall_delta=pass, coco_map_delta=FAIL |
| A1 | -0.100 [-0.200, -0.032] | +6.5 [+0.0, +14.6] | +0.025 [-0.075, +0.100] | bg_fp_reduction=pass, recall_delta=pass, coco_map_delta=pass |
| A2 | +0.333 [+0.199, +0.500] | +6.5 [+1.6, +12.3] | +0.175 [+0.075, +0.300] | bg_fp_reduction=FAIL, recall_delta=pass, coco_map_delta=pass |
| A3 | +0.367 [+0.232, +0.534] | +22.6 [+11.6, +33.3] | +0.125 [+0.025, +0.225] | bg_fp_reduction=FAIL, recall_delta=pass, coco_map_delta=pass |
| A4 | +0.133 [-0.033, +0.333] | +29.0 [+17.2, +41.4] | +0.075 [-0.050, +0.176] | bg_fp_reduction=FAIL, recall_delta=pass, coco_map_delta=pass |
| A5 | -0.133 [-0.234, -0.033] | +0.0 [+0.0, +0.0] | -0.025 [-0.075, +0.000] | bg_fp_reduction=pass, recall_delta=pass, coco_map_delta=pass |

Criteria: bg FP reduction ≥ 30%, recall drop ≤ 2.0 pts, COCO mAP50-95 drop ≤ 1.0 pts. A delta whose CI includes 0 is not distinguishable from noise.
