"""Shared helpers for the SIVP revision experiments.

Everything in this module is pure Python (csv/json/math/random/statistics) so the
frozen-output analyses run even on nodes where NumPy is not installed. NumPy is
used opportunistically for speed when it is importable.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # optional acceleration only
    import numpy as _np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    _np = None

EXITS = (1, 2, 3, 4)
TRANSITIONS = (1, 2, 3)  # k -> k+1
METRICS = ("dice", "iou", "hd95", "assd", "boundary_iou")
DICE_TOL = 0.001
HD95_TOL = 0.5
DIAGONAL_256 = math.hypot(256, 256)  # HD95/ASSD fallback used by bmas.metrics for empty masks

# Values printed in the submitted manuscript. Used only for sanity checks and for
# disambiguating run directories; never copied into result tables.
PAPER_EXIT4_DICE = {
    "UNet": 0.8658, "DeepLabV3": 0.8658, "EffB0UNet": 0.8743,
    "A": 0.8763, "B": 0.8767, "C": 0.8762, "D": 0.8755, "E": 0.8762,
}
PAPER_TABLE5 = {  # MVR, BMVR, E3->E4 Dice viol., E3->E4 HD95 viol. (percent)
    "A": (8.55, 14.38, 9.69, 18.72),
    "B": (8.09, 13.70, 8.64, 18.72),
    "C": (8.41, 13.55, 9.65, 18.64),
    "D": (8.53, 14.41, 12.75, 22.40),
    "E": (6.89, 12.07, 7.75, 16.01),
}
PAPER_GFLOPS_FALLBACK = {1: 1.065, 2: 1.320, 3: 1.787, 4: 8.093}
BMAS_VARIANTS = ("A", "B", "C", "D", "E")
BASELINES = ("UNet", "DeepLabV3", "EffB0UNet", "nnUNet2D")
COMPETITORS = ("MESS", "ADPC")  # same-backbone independent-exit comparators


# ----------------------------------------------------------------------------- IO
def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_csv_dicts(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt_cell(r.get(k, "")) for k in fieldnames})


def _fmt_cell(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.10g}"
    return v


def to_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_md(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


# ------------------------------------------------------------ per-case tables
_COL_PATTERNS = (
    "e{k}_{m}", "exit{k}_{m}", "exit_{k}_{m}", "{m}_e{k}", "{m}_exit{k}", "{m}_exit_{k}", "E{k}_{m}",
)


def detect_exit_columns(fieldnames: Sequence[str]) -> Dict[str, str]:
    """Map canonical names ``e{k}_{metric}`` to the column names present in a CSV."""
    present = set(fieldnames)
    mapping: Dict[str, str] = {}
    for k in EXITS:
        for m in METRICS:
            for pat in _COL_PATTERNS:
                col = pat.format(k=k, m=m)
                if col in present:
                    mapping[f"e{k}_{m}"] = col
                    break
    # Single-exit tables (endpoint baselines): plain metric columns are copied to every exit,
    # the same convention bmas.model uses for static baselines.
    if not any(k.startswith("e1_") for k in mapping):
        for m in METRICS:
            for col in (m, f"final_{m}", f"{m}_final"):
                if col in present:
                    for k in EXITS:
                        mapping.setdefault(f"e{k}_{m}", col)
                    break
    return mapping


def is_per_case_exit_table(path: Path) -> bool:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            mapping = detect_exit_columns(reader.fieldnames or [])
            n = sum(1 for _ in reader)
    except Exception:
        return False
    need = {f"e{k}_dice" for k in EXITS} | {f"e{k}_hd95" for k in EXITS}
    return need.issubset(mapping) and n >= 20


CaseTable = Dict[str, Dict[str, Any]]


def load_case_table(path: str | Path) -> CaseTable:
    """Load a per-case CSV into ``{case_id: {canonical_metric: float, ...meta}}``."""
    rows = read_csv_dicts(path)
    if not rows:
        raise ValueError(f"empty per-case CSV: {path}")
    mapping = detect_exit_columns(list(rows[0].keys()))
    id_col = next((c for c in ("case_id", "id", "case", "image_id", "name") if c in rows[0]), None)
    table: CaseTable = {}
    for idx, r in enumerate(rows):
        cid = str(r[id_col]) if id_col else f"row{idx:05d}"
        rec: Dict[str, Any] = {}
        for canon, col in mapping.items():
            rec[canon] = to_float(r.get(col))
        for meta in ("tumor_code", "tumor_label", "plane_code", "plane_label", "lesion_pixels", "image_path"):
            if meta in r and r[meta] not in ("", "unknown", None):
                rec[meta] = r[meta]
        mapped = set(mapping.values())
        for col, val in r.items():  # keep any other numeric column (e.g. e1_uncert, e34_new_cc)
            if col in mapped or col in rec or col == id_col:
                continue
            fv = to_float(val)
            if fv is not None:
                rec[col] = fv
        rec["_row"] = idx
        table[cid] = rec
    return table


def align_case_ids(tables: Sequence[CaseTable]) -> List[str]:
    """Common case ids across tables. Falls back to row order when ids differ but sizes match."""
    if not tables:
        return []
    common = set(tables[0])
    for t in tables[1:]:
        common &= set(t)
    if len(common) == len(tables[0]) and all(len(t) == len(common) for t in tables):
        return sorted(common, key=lambda c: tables[0][c]["_row"])
    sizes = {len(t) for t in tables}
    if len(sizes) == 1 and len(common) < len(tables[0]):
        raise ValueError(
            "case_id values differ between per-case tables of equal length; refusing to align by row order. "
            "Re-evaluate with a common manifest."
        )
    return sorted(common, key=lambda c: tables[0][c]["_row"])


# ------------------------------------------------------------- statistics
def mean(xs: Iterable[Optional[float]]) -> float:
    vals = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.fmean(vals) if vals else float("nan")


def sd(xs: Iterable[Optional[float]]) -> float:
    vals = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else float("nan"))


def pm(values: Sequence[Optional[float]], digits: int = 3, scale: float = 1.0) -> str:
    vals = [v * scale for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return "n/a"
    m = statistics.fmean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def percentile(values: Sequence[float], q: float) -> float:
    xs = sorted(v for v in values if v is not None and math.isfinite(v))
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def cvar_upper(values: Sequence[float], alpha: float = 0.95) -> float:
    """Mean of the largest (1-alpha) fraction of values (CVaR / expected shortfall)."""
    xs = sorted((v for v in values if v is not None and math.isfinite(v)), reverse=True)
    if not xs:
        return float("nan")
    k = max(1, int(math.ceil(len(xs) * (1.0 - alpha))))
    return statistics.fmean(xs[:k])


def bootstrap_ci_mean(diffs: Sequence[float], n_boot: int = 5000, seed: int = 20261006,
                      level: float = 0.95) -> Tuple[float, float]:
    xs = [d for d in diffs if d is not None and math.isfinite(d)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    if _np is not None:
        rng = _np.random.default_rng(seed)
        arr = _np.asarray(xs, dtype=float)
        idx = rng.integers(0, n, size=(n_boot, n))
        means = arr[idx].mean(axis=1)
        lo, hi = _np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
        return float(lo), float(hi)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += xs[rng.randrange(n)]
        means.append(s / n)
    return percentile(means, 100 * (1 - level) / 2), percentile(means, 100 * (1 + level) / 2)


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_signed_rank(diffs: Sequence[float]) -> Tuple[float, float, int]:
    """Two-sided Wilcoxon signed-rank test (zero differences dropped, tie-corrected
    normal approximation with continuity correction). Returns (W+, p, n_nonzero)."""
    xs = [d for d in diffs if d is not None and math.isfinite(d) and d != 0.0]
    n = len(xs)
    if n == 0:
        return 0.0, 1.0, 0
    order = sorted(range(n), key=lambda i: abs(xs[i]))
    ranks = [0.0] * n
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(xs[order[j + 1]]) == abs(xs[order[i]]):
            j += 1
        r = (i + j + 2) / 2.0
        for t in range(i, j + 1):
            ranks[order[t]] = r
        t_size = j - i + 1
        tie_term += t_size ** 3 - t_size
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, xs) if d > 0)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return w_plus, 1.0, n
    z = (abs(w_plus - mu) - 0.5) / math.sqrt(var)
    return w_plus, min(1.0, 2.0 * _norm_sf(max(z, 0.0))), n


def bh_adjust(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvals[i] * m / rank)
        adj[i] = min(1.0, running)
    return adj


def logistic_fit(X: List[List[float]], y: List[float], ridge: float = 1e-6, iters: int = 50) -> List[float]:
    """Newton/IRLS logistic regression. X must include an intercept column."""
    if _np is not None:
        A = _np.asarray(X, dtype=float)
        t = _np.asarray(y, dtype=float)
        beta = _np.zeros(A.shape[1])
        for _ in range(iters):
            eta = _np.clip(A @ beta, -30, 30)
            p = 1.0 / (1.0 + _np.exp(-eta))
            w = p * (1 - p)
            g = A.T @ (t - p) - ridge * beta
            H = (A * w[:, None]).T @ A + ridge * _np.eye(A.shape[1])
            step = _np.linalg.solve(H, g)
            beta = beta + step
            if float(_np.max(_np.abs(step))) < 1e-8:
                break
        return [float(b) for b in beta]
    p_dim = len(X[0])
    beta = [0.0] * p_dim
    for _ in range(iters):
        g = [-ridge * b for b in beta]
        H = [[(ridge if i == j else 0.0) for j in range(p_dim)] for i in range(p_dim)]
        for row, t in zip(X, y):
            eta = max(-30.0, min(30.0, sum(b * x for b, x in zip(beta, row))))
            p = 1.0 / (1.0 + math.exp(-eta))
            w = p * (1 - p)
            for i in range(p_dim):
                g[i] += (t - p) * row[i]
                wi = w * row[i]
                for j in range(i, p_dim):
                    H[i][j] += wi * row[j]
        for i in range(p_dim):
            for j in range(i):
                H[i][j] = H[j][i]
        step = _solve(H, g)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-8:
            break
    return beta


def _solve(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-15:
            continue
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-15 else 0.0 for i in range(n)]


# ---------------------------------------------------------- regression helpers
def transition_flags(rec: Dict[str, Any], k: int, dice_tol: float = DICE_TOL,
                     hd95_tol: float = HD95_TOL) -> Tuple[Optional[bool], Optional[bool]]:
    d0, d1 = rec.get(f"e{k}_dice"), rec.get(f"e{k+1}_dice")
    h0, h1 = rec.get(f"e{k}_hd95"), rec.get(f"e{k+1}_hd95")
    dv = None if d0 is None or d1 is None else (d1 < d0 - dice_tol)
    hv = None if h0 is None or h1 is None else (h1 > h0 + hd95_tol)
    return dv, hv


def regression_rates(table: CaseTable, ids: Sequence[str], dice_tol: float = DICE_TOL,
                     hd95_tol: float = HD95_TOL) -> Dict[str, float]:
    dv_all, hv_all = [], []
    per_k = {k: ([], []) for k in TRANSITIONS}
    for cid in ids:
        rec = table[cid]
        for k in TRANSITIONS:
            dv, hv = transition_flags(rec, k, dice_tol, hd95_tol)
            if dv is not None:
                dv_all.append(dv)
                per_k[k][0].append(dv)
            if hv is not None:
                hv_all.append(hv)
                per_k[k][1].append(hv)
    out = {
        "mvr": 100.0 * mean([float(x) for x in dv_all]),
        "bmvr": 100.0 * mean([float(x) for x in hv_all]),
    }
    for k in TRANSITIONS:
        out[f"dice_viol_e{k}e{k+1}"] = 100.0 * mean([float(x) for x in per_k[k][0]])
        out[f"hd95_viol_e{k}e{k+1}"] = 100.0 * mean([float(x) for x in per_k[k][1]])
    return out


def parse_brisc_filename(name: str) -> Optional[Dict[str, str]]:
    """Parse ``brisc2025_<split>_<index>_<tumor>_<plane>_<sequence>.<ext>``."""
    m = re.search(r"brisc2025_(train|test)_(\d+)_([a-z]{2})_([a-z]{2})_([A-Za-z0-9]+)", Path(name).name)
    if not m:
        return None
    return {"official_split": m.group(1), "official_index": m.group(2), "tumor_code": m.group(3),
            "plane_code": m.group(4), "sequence": m.group(5)}
