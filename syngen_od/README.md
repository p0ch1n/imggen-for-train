# syngen_od: improving detection models with generated data (feasibility study SOP)

Goal: **without collecting new data**, use image generation / editing models (Qwen-Image, Qwen-Image-Edit) to
reduce false positives (grass bumps → person) and missed detections in the COCO-trained YOLOv12 / RF-DETR models,
and use controlled experiments to decide whether the approach is worth it.

> All steps can run end to end in this environment with mock models (`tools/run_smoke.sh`). In the isolated env you
> only change `configs/pipeline.yaml` (paths, backend) and run the same scripts.

---

## 0. Overall flow

```
                    ┌─ 02 Qwen-Image text-to-image hard negatives ─┐
COCO ─ 01 subset ───┼─ 03 Qwen-Image-Edit remove people (paired) ──┼─ 06 teacher detect ─ 07 auto filter ─ 08 spot check ─ 09 camera degrade ─┐
                    └─ 04 Qwen-Image-Edit style transfer ──────────┘                                                                       │
deploy frames (unlabelled) ─ 06 teacher ─ 10 pseudo labels ─ (05 inpaint insertion, optional) ─────────────────────────────────────────────┤
                                                                                                                                           ▼
                        11 leakage check ─ 12 build arms A0–A5 ─ 13 fine-tune ─ 14 predict ─ 15 evaluate ─ 16 compare + pass/fail
```

| Arm | Training data | Question it answers |
|---|---|---|
| **A0** (control) | COCO subset | Does fine-tuning longer alone already help? (must have) |
| A1 | + hard negatives (02 + 03) | Do false positives go down? |
| A2 | + style transfer (04) | Do misses under night/rain/fog improve? |
| A3 | A1 + A2 | Are the effects additive? |
| A4 | + pseudo-labelled real deployment frames (10) | Non-generative baseline; usually the strongest |
| A5 (optional) | + inpaint insertion (05) | Do small, distant targets improve? |

Rough timeline: preparation 2–3 days, generation 1–2 days, training + evaluation 2–3 days.

---

## 1. Preparation (the only part that takes manual work; nothing can be judged without it)

### 1.1 Real evaluation sets (most important)
Synthetic data cannot be used to evaluate synthetic data. Draw from frames **already captured in production**
(not "new collection"):

| Set | Count | Contents | Location |
|---|---|---|---|
| `deploy_eval` | 200–500 frames | Real frames incl. known misses / false positives, **fully labelled** (YOLO txt, same class order as `classes`) | `data/deploy_eval/images` + `labels` |
| `deploy_bg` | 200+ frames | Frames confirmed to contain **no** targets, especially false-positive scenes (grass, bumps, ...) | `data/deploy_bg/images` (no labels needed) |
| `deploy_unlabeled` | 1k–10k frames | Unlabelled frames (for A4 / insertion only) | `data/deploy_unlabeled/images` |

Rules:
- `deploy_eval` / `deploy_bg` must be **split by camera or by date** from `deploy_unlabeled` (consecutive frames are
  near-duplicates, which leak and inflate A4's score). Step 11 checks this automatically.
- Label `deploy_eval` exhaustively. Every missed label later counts as a "false positive".
- File names must be unique within each set (predictions are keyed by file name).

### 1.2 Models and packages (offline transfer)
On a machine with internet access:
```bash
PY_VER=3.10 CUDA_TAG=cu124 bash offline/prepare_bundle.sh /path/to/bundle
```
This downloads wheels, Qwen-Image, Qwen-Image-Edit-2509, Lightning LoRA, Grounding DINO, FLUX.1-Fill (optional),
RF-DETR / YOLO weights and this code, and writes SHA256SUMS.
For FLUX.1-Fill-dev, accept the license on HF and `huggingface-cli login` first; skip it with `SKIP_FLUX=1` if you don't need it.

In the isolated env:
```bash
cd bundle && sha256sum -c SHA256SUMS
python -m venv venv && source venv/bin/activate
pip install --no-index --find-links wheels torch torchvision
pip install --no-index --find-links wheels -r code/syngen_od/requirements.txt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mv models/* /data/syngen_od/models/          # matches paths.models in the config
cp <production YOLOv12 weights> /data/syngen_od/models/prod_yolov12s.pt
```

> Pip wheels are platform-sensitive. If cross-platform `pip download` fails, the most reliable fallback is to build a
> docker image on a matching machine and transfer the image with `docker save`.

### 1.3 Edit the config `configs/pipeline.yaml`
Required changes:
- `root`: working directory
- `classes`: **must match the production model's class names and order exactly** (if production is the 80-class
  model, list all 80, otherwise the head is re-initialised)
- `train.weights`: production weights; `train.args`: same imgsz / batch as production training
- `eval.conf_thr`: the production confidence threshold
- `configs/prompts.yaml` → `negatives.confusers`: **add the false-positive objects you've actually seen** (most important)

### 1.4 Pre-flight check
```bash
python scripts/00_check_env.py --config configs/pipeline.yaml
```
Every FAIL must be fixed. If VRAM < 40GB, set `generation.*.offload: model`.

---

## 2. Data preparation and generation

> All generation scripts are **resumable** (re-running skips finished items) and support multi-GPU sharding:
> `CUDA_VISIBLE_DEVICES=k python scripts/0X_... --shard k --num-shards N` (one process per GPU).
> Try `--limit 20` first, check the images look right, then run everything.

### Step 01 COCO subset
```bash
python scripts/01_coco_subset.py --config configs/pipeline.yaml
```
Check: the printed image / box counts are reasonable; `data/coco_subset/images/train` has symlinks.

### Step 02 Hard negatives (text-to-image)
```bash
python scripts/02_gen_neg_t2i.py --config configs/pipeline.yaml --limit 20   # trial
python scripts/02_gen_neg_t2i.py --config configs/pipeline.yaml
```
Check: open 20 images. Do they look like the false-positive scenes? **Did people or cars appear anyway?** If the
view or style is off, edit `views` / `conditions` in prompts.yaml.

### Step 03 People removal (paired negatives)
```bash
python scripts/03_gen_remove.py --config configs/pipeline.yaml --limit 20
python scripts/03_gen_remove.py --config configs/pipeline.yaml
```
Check: are people removed cleanly (no ghosting / half bodies)? Is the rest of the scene unchanged?

### Step 04 Style transfer (labels preserved)
```bash
python scripts/04_gen_style.py --config configs/pipeline.yaml --limit 20
python scripts/04_gen_style.py --config configs/pipeline.yaml [--styles night_ir rain fog]
```
Check: **are objects still in the same place**? Qwen-Image-Edit sometimes moves people, adds people or crops the
frame. Step 07 filters these, but if more than 30% are bad at the trial stage, rewrite that style's prompt first.

### Step 05 (optional) Insertion into deployment backgrounds
Requires step 10 first (backgrounds need trustworthy labels).
```bash
python scripts/05_gen_insert.py --config configs/pipeline.yaml
```

### Throughput reference (A100 80GB, bf16, no offload; actual numbers vary by hardware)
| Model | Settings | Per image |
|---|---|---|
| Qwen-Image 1664×928 | 50 steps | ~30–60 s |
| Qwen-Image + Lightning | 8 steps, true_cfg 1.0 | ~5–10 s |
| Qwen-Image-Edit-2509 ~1MP | 40 steps | ~30–60 s |

2,000 images × 50 s ≈ 28 GPU hours. **Lightning is recommended** (see the comments in the config). Compare a few
images with and without it before switching.

---

## 3. Quality control (automatic + manual)

### Step 06 + 07 Teacher detection and automatic filtering
```bash
for T in neg_t2i remove style; do
  python scripts/06_teacher_detect.py --config configs/pipeline.yaml --task $T
  python scripts/07_filter.py --config configs/pipeline.yaml --task $T
done
```
Filtering rules:

| Task | Rejected when |
|---|---|
| neg_t2i | Teacher detects any target class (conf ≥ 0.25) → the generated image contains a person / car |
| remove | The removed class can still be detected; the rest of the scene changed (structure similarity < threshold); a kept object disappeared |
| style | Scene changed; an object moved / disappeared (both teacher and structure check fail); an unlabelled high-confidence object appeared (hallucination) |
| insert | The inserted object isn't detected by the teacher inside the mask (passing boxes are replaced with the teacher's tight box) |

"Structure similarity" is the correlation of gradient structure after contrast normalisation. It is insensitive to
night / colour / fog changes but very sensitive to objects moving or disappearing. So in hard samples where the
teacher misses an object at night, the box can still be kept as long as the structure didn't change; those are exactly
the samples we want.

Output: `kept.txt`, `filter_report.csv` (per-image scores, use them to tune thresholds), `filter_summary.json` (pass
rate per style / per confuser).
**Check:** if a style's pass rate is < 30%, drop it or rewrite its prompt; if the overall pass rate is > 95%, the
thresholds are probably too loose. Go through `filter_report.csv` and look at the images near the threshold.

### Step 08 Manual spot check
```bash
python scripts/08_review.py --config configs/pipeline.yaml --task style --frac 0.05 --pairs
# look at <task>/review/sheet_*.png, write bad image numbers into review/rejects.txt (comma- or newline-separated)
python scripts/08_review.py --config configs/pipeline.yaml --task style --apply
```
What to look for: wrong box positions, unlabelled people / cars, removal residue, obvious AI artifacts.
**If more than 10% of the spot-checked sample is bad, go back and fix 07's thresholds or the prompts; don't rely on
manual review to clean up.**

### Step 09 Camera-like degradation
```bash
for T in neg_t2i remove style; do python scripts/09_degrade.py --config configs/pipeline.yaml --task $T; done
```
Generated images are too clean; without this the model can learn "clean image = no person" as a shortcut. Tune the
`degrade` ranges to your camera quality (compression, resolution).

---

## 4. Pseudo labels (A4, the non-generative baseline)
```bash
python scripts/06_teacher_detect.py --config configs/pipeline.yaml \
    --images /data/syngen_od/data/deploy_unlabeled/images --out-dir /data/syngen_od/data/pseudo
python scripts/10_pseudo_label.py --config configs/pipeline.yaml
```
Frames with dets in the ambiguous band [0.2, 0.5) are dropped (YOLO has no ignore regions). Spot-check
`data/pseudo` the same way.

---

## 5. Experiments

### Step 11 Leakage check (required)
```bash
python scripts/11_dedupe_check.py --config configs/pipeline.yaml   # add --remove to drop offending images automatically
```
If you see `<-- LEAK`, A4's (and A5's) results are not trustworthy. Split pseudo-label sources by camera / date and redo.

### Step 12 Build the datasets
```bash
python scripts/12_build_experiments.py --config configs/pipeline.yaml
```
Check each arm's `composition.json`: added data is best kept at **10–30%** of the total. Too much synthetic data
hurts COCO / real-world performance.

### Step 13 Training
```bash
python scripts/13_train.py --config configs/pipeline.yaml --dry-run          # check parameters
# multiple GPUs: one arm per GPU
for i in 0 1 2 3 4; do A=(A0 A1 A2 A3 A4); \
  nohup python scripts/13_train.py --config configs/pipeline.yaml --arms ${A[$i]} --device $i > train_${A[$i]}.log 2>&1 & done
```
Key points:
- Every arm uses **the same** starting weights, epochs, lr and seed. The only difference is the data.
- `optimizer` must be set explicitly (Ultralytics' `auto` ignores lr0).
- Fine-tune with a low lr (lr0 0.001–0.003). Going too high causes forgetting.
- RF-DETR: `python tools/export_rfdetr_coco.py --config ... --arm A1 --out .../rfdetr/A1`, then train with the
  rfdetr package (example in the script's header). For prediction use `14_predict.py --backend rfdetr` and verify
  the class mapping.

### Steps 14–16 Evaluation and comparison
```bash
python scripts/14_predict.py  --config configs/pipeline.yaml    # prod + every trained arm × deploy_eval / deploy_bg / coco_val
python scripts/15_evaluate.py --config configs/pipeline.yaml
python scripts/16_compare.py  --config configs/pipeline.yaml    # -> results/report.md
```
The report includes:
- mAP50-95 / mAP50 (COCO-style; implementation verified identical to pycocotools)
- At the production threshold `conf_thr`: per-class recall, small-object recall, FP/image
- `deploy_bg` FP/image and the fraction of frames with any false positive (the main false-positive metric)
- Paired-bootstrap 95% CI of each difference vs A0. **A delta whose CI includes 0 is noise.**
- Pass/fail against `criteria`

---

## 6. Interpreting results / decisions

| Result | Conclusion / next step |
|---|---|
| A1 bg FP ↓ ≥ 30% (CI excludes 0), recall roughly unchanged | Hard negatives work → add more confuser types and scale up |
| A1 FP ↓ but recall ↓ clearly | Too many negatives or too similar to real targets → lower the ratio or remove human-like objects from confusers |
| A2 small / night recall ↑ | Style transfer works → focus on the styles with the most gain (per-style comparison: build a separate arm per style) |
| A0 itself improves a lot vs prod | Part of the gain comes from "re-fine-tuning on the class subset"; always compare against A0 |
| A4 ≫ A1/A2 | Real-domain data is the key → prioritise pseudo labels; generated data is a supplement |
| Nothing beats A0 | Check filter quality (08 reject rate), degradation, data ratio; then consider the evaluation set may be too small |
| COCO mAP drops > 1 pt | Too much synthetic data / lr too high → lower the ratio or lr |

Things to bring back after the study (they contain no images, so they can leave the isolated env):
`results/report.md`, `results/report.json`, each task's `filter_summary.json`, `08` reject rates, `composition.json`.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA OOM during generation | `offload: model` → `sequential`; or lower `sizes` |
| `DiffusionPipeline` can't find the class | diffusers too old; needs a version with `QwenImageEditPlusPipeline` (00 checks this) |
| Edited output size differs from input | Expected; the script resizes back to the original size so boxes stay aligned |
| Edits are mostly black / blank | `is_degenerate` filters these automatically; if frequent, check dtype (use bf16, not fp16) |
| Very low 07 pass rate | Look at `filter_report.csv` reasons; for `scene_changed`, lower `global_struct_min` or strengthen "keep" in the prompt |
| RF-DETR class mapping is off | Set `eval.predict.id_to_name` or `id_offset`, and check a few predictions visually |
| Many predictions for prod, A0 drops sharply | `classes` doesn't match the production model (order / count) |

---

## 8. Project layout
```
configs/pipeline.yaml   main config (paths, thresholds, experiment arms)
configs/prompts.yaml    all prompts (confusers / styles / insertion objects)
configs/smoke.yaml      smoke-test config (mock models)
scripts/00-16           pipeline steps (see above)
syngen/                 shared modules: metrics, filter checks, generation/detection backends
tools/run_smoke.sh      end-to-end test (CPU, fake data)
tools/make_fake_data.py fake data
tools/export_rfdetr_coco.py
offline/prepare_bundle.sh  offline package + model bundling
tests/                  unit tests (incl. mAP vs pycocotools)
```

Testing (any machine):
```bash
pip install numpy pillow pyyaml pytest pycocotools
python -m pytest -q tests
bash tools/run_smoke.sh /tmp/smoke --no-train     # no torch needed
bash tools/run_smoke.sh /tmp/smoke                # with ultralytics: really trains for 1 epoch
```
