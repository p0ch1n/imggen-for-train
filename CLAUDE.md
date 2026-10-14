# imggen-for-train

Feasibility study: use generative image models (Qwen-Image / Qwen-Image-Edit) to synthesize training
data that reduces false positives and missed detections in a COCO-trained object detector deployed on
fixed CCTV cameras (target domain: semiconductor fab interior, office areas, and perimeter/yard), without
collecting new real data (the production training environment has no external data access).

The actual project lives in **`syngen_od/`** — read `syngen_od/README.md` first, it's the full SOP.
`syngen_od/report.html` is a running feasibility report (background, method, SOTA survey, results so far)
— open it in a browser, keep it updated as the study progresses rather than writing a new one from scratch.

## Layout
- `syngen_od/` — the pipeline: `configs/` (pipeline.yaml = real GPU run, local_cpu.yaml = this-machine CPU
  smoke-test harness, smoke.yaml = mock-everything unit test), `scripts/00`-`16` (the pipeline steps),
  `syngen/` (shared modules), `report.html` (living report), `_local_run/` (local demo outputs — small,
  committed; the actual generated-image *datasets* under `data/gen/*` are scratch, not committed).
- `models/`, `hf_cache/`, `syngen_local/` — downloaded model weights / HF cache / COCO annotations.
  **Not in git** (see `.gitignore`): multi-GB, re-downloadable, and several individual files exceed
  GitHub's 100MB hard file-size limit. Re-fetch per `syngen_od/README.md` / `offline/prepare_bundle.sh`.

## Environment gotcha (read before running anything generative here)
`/home/kasm-user/Personal_Data` is a **gcsfuse** mount. Diffusers reads `.safetensors` via `mmap`, which
on gcsfuse turns into random small reads at ~1MB/s (a 2.5GB model can take ~50 minutes to "load").
Sequential copies are fine (~100MB/s). Rule: **before running any model inference, `cp` the model weights
and build any Python venv on local disk** (e.g. `/home/kasm-user/.cache/models/...`,
`/home/kasm-user/.venvs/...`), never read/write them in place on `Personal_Data`. Only read/write small
files (code, configs, a handful of result images) directly on the gcsfuse path.

## This machine has no GPU
CPU-only (2 vCPU). `configs/local_cpu.yaml` is a from-scratch config (not `extends: pipeline.yaml` —
Qwen's `true_cfg_scale` kwarg isn't accepted by non-Qwen pipelines) that swaps in whatever small model is
locally available (currently `stabilityai/sd-turbo`, CPU-feasible) as a stand-in for Qwen-Image, purely to
smoke-test the pipeline plumbing and sanity-check `configs/prompts.yaml` wording — **not** to judge final
data quality. The real run (Qwen-Image / Qwen-Image-Edit-2509, RF-DETR teacher) needs a GPU box; see
`syngen_od/README.md` section 1.2 (`offline/prepare_bundle.sh`) for the offline-transfer workflow into an
isolated GPU environment.

## Prompts / confusers
`configs/prompts.yaml` is written for the actual target domain (fab interior / office / perimeter), not
a generic placeholder. `negatives.confusers` is a **guessed starting list** (cleanroom suits, gas
cylinders, AGVs, etc.) — replace/extend it with objects that have *actually* caused false positives on
the real cameras; that's more valuable than anything inferred here. When adding entries, make sure a
confuser is something that visually resembles a target class but **is not one** — don't describe the
target class itself in a new scene (e.g. "a bicycle parking area" generates actual bicycles, which is a
real `bicycle`-class object, not a look-alike — this mistake happened once already, see report.html §6.3).

## Workflow reminders
- Every generation/filter/degrade script is resumable and shardable (`--limit N` for a quick trial first).
- `09_degrade.py`'s optional `p_fisheye` knob warps pixel positions — only safe for tasks with no real
  boxes to invalidate (`neg_t2i`). Never enable it for `remove`/`style` without also remapping the boxes.
- Never commit generated *datasets* (`syngen_od/_local_run/data/gen/**` beyond the small demo images
  already there, or any future large batch) — those are scratch outputs, not source.
