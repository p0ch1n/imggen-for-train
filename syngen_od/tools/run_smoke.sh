#!/usr/bin/env bash
# End-to-end smoke test with mock generators/teacher and fake data (CPU only, a few minutes).
# Usage: bash tools/run_smoke.sh <empty workdir> [--no-train]
set -euo pipefail
cd "$(dirname "$0")/.."
export SMOKE_ROOT="$(realpath -m "${1:?workdir}")"
NO_TRAIN="${2:-}"
PY="${PYTHON:-python}"
C=configs/smoke.yaml
rm -rf "$SMOKE_ROOT" && mkdir -p "$SMOKE_ROOT"

$PY tools/make_fake_data.py --root "$SMOKE_ROOT"
$PY scripts/00_check_env.py --config $C --skip-models
$PY scripts/01_coco_subset.py --config $C

# generation (two shards for neg_t2i to exercise sharding)
$PY scripts/02_gen_neg_t2i.py --config $C --shard 0 --num-shards 2
$PY scripts/02_gen_neg_t2i.py --config $C --shard 1 --num-shards 2
$PY scripts/03_gen_remove.py --config $C
$PY scripts/04_gen_style.py --config $C

# pseudo labels (needed as insert backgrounds and for A4)
$PY scripts/06_teacher_detect.py --config $C --images "$SMOKE_ROOT/data/deploy_unlabeled/images" --out-dir "$SMOKE_ROOT/data/pseudo"
$PY scripts/10_pseudo_label.py --config $C
$PY scripts/05_gen_insert.py --config $C

for T in neg_t2i remove style insert; do
  $PY scripts/06_teacher_detect.py --config $C --task $T
  $PY scripts/07_filter.py --config $C --task $T
done
$PY scripts/08_review.py --config $C --task style --frac 0.5 --pairs
echo "0, 2" > "$SMOKE_ROOT/data/gen/style/review/rejects.txt"
$PY scripts/08_review.py --config $C --task style --apply
for T in neg_t2i remove style insert; do $PY scripts/09_degrade.py --config $C --task $T; done
$PY scripts/09_degrade.py --config $C --task style   # second run must be a no-op

$PY scripts/11_dedupe_check.py --config $C
$PY scripts/12_build_experiments.py --config $C
$PY tools/export_rfdetr_coco.py --config $C --arm A1 --out "$SMOKE_ROOT/rfdetr/A1"
$PY scripts/13_train.py --config $C --dry-run

if [ "$NO_TRAIN" != "--no-train" ]; then
  (cd "$SMOKE_ROOT" && $PY "$OLDPWD/scripts/13_train.py" --config "$OLDPWD/$C")
  (cd "$SMOKE_ROOT" && $PY "$OLDPWD/scripts/14_predict.py" --config "$OLDPWD/$C")
else
  # no ultralytics: fake predictions from the mock detector so 15/16 still run
  $PY scripts/14_predict.py --config $C --name A0 --weights none --backend mock
  $PY scripts/14_predict.py --config $C --name A1 --weights none --backend mock
fi
$PY scripts/15_evaluate.py --config $C
$PY scripts/16_compare.py --config $C
echo "SMOKE TEST OK -> $SMOKE_ROOT/results/report.md"
