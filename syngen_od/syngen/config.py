"""YAML config loading with ${a.b} and ${env:VAR} interpolation."""
import os
import re
import sys
from pathlib import Path

import yaml

_VAR = re.compile(r"\$\{([^}]+)\}")


def _lookup(cfg, dotted):
    if dotted.startswith("env:"):
        return os.environ[dotted[4:]]
    cur = cfg
    for k in dotted.split("."):
        cur = cur[k]
    return cur


def _resolve(node, cfg):
    if isinstance(node, dict):
        return {k: _resolve(v, cfg) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, cfg) for v in node]
    if isinstance(node, str):
        for _ in range(10):
            m = _VAR.fullmatch(node)
            if m:  # whole-string reference keeps the referenced type (list, int, ...)
                node = _lookup(cfg, m.group(1))
                if not isinstance(node, str):
                    return _resolve(node, cfg)
                continue
            new = _VAR.sub(lambda m: str(_lookup(cfg, m.group(1))), node)
            if new == node:
                break
            node = new
    return node


def _deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "extends" in raw:  # `extends: other.yaml` (relative to this file): deep-merge this file over it
        base = _load_raw(Path(path).parent / raw.pop("extends"))
        raw = _deep_merge(base, raw)
    return raw


def load_config(path):
    cfg = _load_raw(path)
    for _ in range(5):  # resolve nested references
        cfg = _resolve(cfg, cfg)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def get(cfg, dotted, default=None):
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def project_root():
    return Path(__file__).resolve().parent.parent


def add_project_to_path():
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_path(p):
    """Relative paths are taken from the project root (the syngen_od directory)."""
    p = Path(p)
    return p if p.is_absolute() else project_root() / p


def load_prompts(cfg):
    with open(resolve_path(cfg["prompts_file"]), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
