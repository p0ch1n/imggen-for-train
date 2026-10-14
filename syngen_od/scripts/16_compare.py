#!/usr/bin/env python3
"""16 Compare all models against the reference arm and apply the feasibility criteria.

Writes <results>/report.md and <results>/report.json. Deltas on the deployment sets come with paired-bootstrap
95% CIs over images (same resampled images for both models), so small eval sets don't produce false conclusions.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syngen.common import base_parser, load  # noqa: E402
from syngen.records import read_json, write_json  # noqa: E402


def per_image_arrays(m, nc):
    """-> image keys, (I, C, 3) array of [tp, fp, fn]"""
    pi = m["op"]["per_image"]
    keys = sorted(pi)
    return keys, np.array([[pi[k][str(c)] for c in range(nc)] for k in keys], dtype=np.float64).reshape(-1, nc, 3)


def stat(arr, kind, cls=None):
    """arr (I,C,3). kind: recall | fp_per_image."""
    a = arr if cls is None else arr[:, [cls]]
    tp, fp, fn = a[..., 0].sum(), a[..., 1].sum(), a[..., 2].sum()
    if kind == "recall":
        return tp / (tp + fn) if tp + fn else np.nan
    return fp / max(len(arr), 1)


def paired_bootstrap(a, b, kind, n, seed, cls=None):
    """CI of stat(b) - stat(a) where a/b are aligned per-image arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    diffs = np.array([stat(b[i], kind, cls) - stat(a[i], kind, cls) for i in idx])
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return None
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def fmt(x, pct=False, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x * 100:+.1f}" if sign and pct else f"{x * 100:.1f}" if pct else f"{x:+.3f}" if sign else f"{x:.3f}"
    return s


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg = load(args)
    ev, cr = cfg["eval"], cfg["criteria"]
    classes = cfg["classes"]
    nc = len(classes)
    root = Path(cfg["paths"]["results"])
    M = {}
    for f in root.glob("*/*/metrics.json"):
        m = read_json(f)
        M.setdefault(m["model"], {})[m["set"]] = m
    if not M:
        sys.exit(f"no metrics.json under {root}; run 15_evaluate.py first")
    ref = ev["reference"]
    if ref not in M:
        sys.exit(f"reference model {ref} has no metrics; available: {sorted(M)}")
    order = [k for k in ["prod"] + list(cfg["experiments"]["arms"]) if k in M] + sorted(set(M) - {"prod"} - set(
        cfg["experiments"]["arms"]))
    B, seed = ev.get("bootstrap", 1000), cfg["seed"]

    def g(model, s, *path):
        cur = M.get(model, {}).get(s)
        for k in path:
            if cur is None:
                return None
            cur = cur.get(k) if isinstance(cur, dict) else None
        return cur

    report = {"reference": ref, "conf_thr": ev["conf_thr"], "models": {}}
    for name in order:
        r = {"deploy_map50_95": g(name, "deploy_eval", "ap", "map50_95"),
             "deploy_map50": g(name, "deploy_eval", "ap", "map50"),
             "coco_map50_95": g(name, "coco_val", "ap", "map50_95"),
             "bg_fp_per_image": g(name, "deploy_bg", "op", "fp_per_image_all"),
             "bg_frac_images_with_fp": g(name, "deploy_bg", "op", "frac_images_with_fp_all"),
             "deploy_fp_per_image": g(name, "deploy_eval", "op", "fp_per_image_all"),
             "recall": {}, "recall_small": {}}
        for c in range(nc):
            r["recall"][classes[c]] = g(name, "deploy_eval", "op", "per_class", str(c), "recall")
            r["recall_small"][classes[c]] = g(name, "deploy_eval", "op", "recall_by_size", str(c), "small")
        if "deploy_eval" in M[name]:
            _, arr = per_image_arrays(M[name]["deploy_eval"], nc)
            r["recall_all"] = stat(arr, "recall")
        # paired comparisons vs reference
        ci = {}
        for s, kind in [("deploy_bg", "fp_per_image"), ("deploy_eval", "recall"), ("deploy_eval", "fp_per_image")]:
            if name != ref and s in M[name] and s in M[ref]:
                ka, a = per_image_arrays(M[ref][s], nc)
                kb, b = per_image_arrays(M[name][s], nc)
                if ka == kb:
                    ci[f"{s}.{kind}"] = paired_bootstrap(a, b, kind, B, seed)
        r["delta_ci95"] = ci
        report["models"][name] = r

    # criteria
    R = report["models"][ref]
    for name, r in report["models"].items():
        if name == ref:
            continue
        checks = {}
        if r["bg_fp_per_image"] is not None and R["bg_fp_per_image"] is not None:
            ci = r["delta_ci95"].get("deploy_bg.fp_per_image")
            if R["bg_fp_per_image"] > 0:
                red = 1 - r["bg_fp_per_image"] / R["bg_fp_per_image"]
                ok = red >= cr["bg_fp_reduction_min"]
            else:  # reference has no FP at all: nothing to reduce, only require no new FP
                red, ok = None, r["bg_fp_per_image"] == 0
            checks["bg_fp_reduction"] = {"value": red, "pass": ok, "significant": bool(ci and ci[1] < 0)}
        if r.get("recall_all") is not None and R.get("recall_all") is not None:
            d = r["recall_all"] - R["recall_all"]
            ci = r["delta_ci95"].get("deploy_eval.recall")
            checks["recall_delta"] = {"value": d, "pass": d >= -cr["recall_drop_max"],
                                      "significant": bool(ci and (ci[0] > 0 or ci[1] < 0))}
        if r["coco_map50_95"] is not None and R["coco_map50_95"] is not None:
            d = r["coco_map50_95"] - R["coco_map50_95"]
            checks["coco_map_delta"] = {"value": d, "pass": d >= -cr["coco_map_drop_max"]}
        r["criteria"] = checks
        r["feasible"] = bool(checks) and all(c["pass"] for c in checks.values())

    write_json(root / "report.json", report)

    # markdown
    L = ["# Feasibility report\n", f"reference = **{ref}**, conf_thr = {ev['conf_thr']}, "
                                     f"bootstrap = {B} (95% CI of paired difference vs reference)\n"]
    hdr = ["model", "deploy mAP50-95", "deploy mAP50", "recall@thr"] + [f"recall {c}" for c in classes] + \
          ["recall small " + classes[0], "deploy FP/img", "bg FP/img", "bg imgs w/ FP", "COCO mAP50-95", "feasible"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for name in order:
        r = report["models"][name]
        cells = [f"**{name}**" if name == ref else name, fmt(r["deploy_map50_95"]), fmt(r["deploy_map50"]),
                 fmt(r.get("recall_all"), pct=True)] + [fmt(r["recall"][c], pct=True) for c in classes] + \
                [fmt(r["recall_small"][classes[0]], pct=True), fmt(r["deploy_fp_per_image"]), fmt(r["bg_fp_per_image"]),
                 fmt(r["bg_frac_images_with_fp"], pct=True), fmt(r["coco_map50_95"]),
                 "ref" if name == ref else ("✅" if r.get("feasible") else "❌")]
        L.append("| " + " | ".join(cells) + " |")
    L.append("\n## Deltas vs reference (95% CI)\n")
    L.append("| model | bg FP/img Δ | deploy recall Δ | deploy FP/img Δ | criteria |")
    L.append("|---|---|---|---|---|")
    for name in order:
        if name == ref:
            continue
        r = report["models"][name]
        def d(key, base_key, pct=False):
            v, b = r.get(base_key), R.get(base_key)
            ci = r["delta_ci95"].get(key)
            if v is None or b is None:
                return "–"
            s = fmt(v - b, pct=pct, sign=True)
            return s + (f" [{fmt(ci[0], pct=pct, sign=True)}, {fmt(ci[1], pct=pct, sign=True)}]" if ci else "")
        crit = ", ".join(f"{k}={'pass' if v['pass'] else 'FAIL'}" for k, v in r.get("criteria", {}).items())
        L.append(f"| {name} | {d('deploy_bg.fp_per_image', 'bg_fp_per_image')} | "
                 f"{d('deploy_eval.recall', 'recall_all', pct=True)} | "
                 f"{d('deploy_eval.fp_per_image', 'deploy_fp_per_image')} | {crit} |")
    L.append(f"\nCriteria: bg FP reduction ≥ {cr['bg_fp_reduction_min']:.0%}, recall drop ≤ "
             f"{cr['recall_drop_max'] * 100:.1f} pts, COCO mAP50-95 drop ≤ {cr['coco_map_drop_max'] * 100:.1f} pts. "
             "A delta whose CI includes 0 is not distinguishable from noise.")
    (root / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
