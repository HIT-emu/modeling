
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contour / bootstrap / profile-likelihood diagnostics for one battery discharge curve.

What this script does:
- reads a CSV with columns test_id, brand_model, info, measurements, discharge_curve
- fits one nonlinear model (Nernst-like) with least_squares
- saves the main fit plot
- saves pairwise RSS surface contour plots (PNG + PDF)
- runs residual bootstrap and saves parameter samples
- runs profile likelihood for each parameter and saves plots + CI table
- saves one extra visualization that highlights the near-equivalent solution region
  using the most correlated parameter pair and bootstrap cloud overlay

By default it processes only the first row (one battery) so it is easy to present.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.stats import chi2, t as student_t


# ------------------------------
# Basic helpers
# ------------------------------
def parse_jsonish(value: Any, default: Any):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {"null", "none", "nan"}:
            return default
        try:
            return json.loads(s)
        except Exception:
            return default
    return default


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return np.nan
        if isinstance(x, str) and x.strip().lower() in {"", "null", "none", "nan"}:
            return np.nan
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def safe_float_array(lst: Sequence[Any]) -> np.ndarray:
    return np.array([safe_float(x) for x in lst], dtype=float)


def sanitize_filename(name: str, max_len: int = 120) -> str:
    name = str(name)
    name = re.sub(r"[^\w\s.-]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name).strip("._ ")
    return (name or "item")[:max_len]


def choose_voltage_series(discharge_curve: Dict[str, Any], preferred_key: Optional[str] = None) -> Tuple[str, str, List[Any], List[Any]]:
    if not isinstance(discharge_curve, dict):
        raise ValueError("discharge_curve is not a dict")

    candidates: List[Tuple[str, str, int, float, List[Any], List[Any]]] = []

    def score_series(time_list: List[Any], volt_list: List[Any]) -> Tuple[int, float]:
        t = safe_float_array(time_list)
        v = safe_float_array(volt_list)
        mask = np.isfinite(t) & np.isfinite(v)
        n = int(mask.sum())
        if n == 0:
            return 0, np.inf
        vrange = float(np.nanmax(v[mask]) - np.nanmin(v[mask])) if n > 1 else 0.0
        return n, vrange

    if preferred_key and preferred_key in discharge_curve:
        obj = discharge_curve[preferred_key]
        if isinstance(obj, dict):
            time_list = obj.get("time", discharge_curve.get("time", []))
            for subkey in ("ua", "ub", "u", "voltage", "V", "v"):
                if subkey in obj:
                    vlist = obj.get(subkey, [])
                    n, vr = score_series(time_list, vlist)
                    candidates.append((preferred_key, subkey, n, vr, time_list, vlist))
            for k, val in obj.items():
                if k != "time" and isinstance(val, list):
                    n, vr = score_series(time_list, val)
                    candidates.append((preferred_key, k, n, vr, time_list, val))

    if "time" in discharge_curve:
        time_list = discharge_curve.get("time", [])
        for subkey in ("ua", "ub", "u", "voltage", "V", "v", "data1", "data2"):
            if subkey not in discharge_curve:
                continue
            val = discharge_curve.get(subkey)
            if isinstance(val, list):
                n, vr = score_series(time_list, val)
                candidates.append(("top", subkey, n, vr, time_list, val))
            elif isinstance(val, dict):
                t2 = val.get("time", time_list)
                for vk in ("ua", "ub", "u", "voltage", "V", "v"):
                    if vk in val:
                        v2 = val.get(vk, [])
                        n, vr = score_series(t2, v2)
                        candidates.append((subkey, vk, n, vr, t2, v2))
                for k, vv in val.items():
                    if k != "time" and isinstance(vv, list):
                        n, vr = score_series(t2, vv)
                        candidates.append((subkey, k, n, vr, t2, vv))
        for k, val in discharge_curve.items():
            if k != "time" and isinstance(val, list):
                n, vr = score_series(time_list, val)
                candidates.append(("top", k, n, vr, time_list, val))

    if not candidates:
        for k, val in discharge_curve.items():
            if isinstance(val, dict) and "time" in val:
                t2 = val.get("time", [])
                for vk, vv in val.items():
                    if vk != "time" and isinstance(vv, list):
                        n, vr = score_series(t2, vv)
                        candidates.append((k, vk, n, vr, t2, vv))

    if not candidates:
        raise ValueError("Не удалось найти пригодную кривую напряжения в discharge_curve")

    candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
    vk, subk, _, _, time_list, volt_list = candidates[0]
    return vk, subk, time_list, volt_list


def extract_capacity_mAh(measurements: Any, info: Any = None) -> Optional[float]:
    meas = parse_jsonish(measurements, default=[])
    if isinstance(meas, dict):
        meas = [meas]

    vals = []
    if isinstance(meas, list):
        for m in meas:
            if not isinstance(m, dict):
                continue
            for key in ("Средняя", "Ёмкость, мАч", "Емкость, мАч", "capacity_mAh", "capacity"):
                if key in m:
                    v = safe_float(m.get(key))
                    if np.isfinite(v):
                        vals.append(v)
                        break
    if vals:
        return float(np.mean(vals))

    info_obj = parse_jsonish(info, default={})
    if isinstance(info_obj, dict):
        for key in ("capacity_mAh", "Ёмкость, мАч", "Емкость, мАч"):
            if key in info_obj:
                v = safe_float(info_obj.get(key))
                if np.isfinite(v):
                    return float(v)
    return None


def infer_element_type(row: pd.Series) -> str:
    if "element_type" in row.index and isinstance(row.get("element_type"), str):
        et = row.get("element_type").strip().lower()
        if et in {"battery", "accumulator"}:
            return et

    if "is_accumulator" in row.index:
        try:
            if int(row.get("is_accumulator")) == 1:
                return "accumulator"
            if int(row.get("is_battery")) == 1:
                return "battery"
        except Exception:
            pass

    text = f"{row.get('brand_model', '')} {row.get('info', '')}".lower()
    markers = [
        "аккум", "accu", "accumulator", "nimh", "ni-mh", "ni cd", "ni-cd", "nicd",
        "recyko", "eneloop", "li-ion", "lithium-ion", "battery pack", "charge",
    ]
    for m in markers:
        if m in text:
            return "accumulator"
    return "battery"


def smooth_voltage(volt_arr: np.ndarray) -> np.ndarray:
    n = len(volt_arr)
    if n < 7:
        return volt_arr.copy()
    window = max(5, min(21, (n // 20) * 2 + 1))
    if window >= n:
        window = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window < 5:
        return volt_arr.copy()
    try:
        return savgol_filter(volt_arr, window_length=window, polyorder=3)
    except Exception:
        return volt_arr.copy()


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def param_confint(params: np.ndarray, cov: np.ndarray, dof: int, alpha: float = 0.05) -> np.ndarray:
    se = np.sqrt(np.clip(np.diag(np.asarray(cov, dtype=float)), 0.0, np.inf))
    tcrit = float(student_t.ppf(1.0 - alpha / 2.0, dof))
    return np.column_stack([params - tcrit * se, params + tcrit * se])


def variance_confint_from_residuals(resid: np.ndarray, p: int, alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    n = len(resid)
    dof = max(n - p, 1)
    s2 = float(np.sum(resid ** 2) / dof)
    lo = dof * s2 / chi2.ppf(1.0 - alpha / 2.0, dof)
    hi = dof * s2 / chi2.ppf(alpha / 2.0, dof)
    return s2, (float(lo), float(hi))


def add_param_columns(row_dict: Dict[str, Any], prefix: str, param_names: List[str], params: np.ndarray, ci: np.ndarray):
    for name, val, (lo, hi) in zip(param_names, params, ci):
        row_dict[f"{prefix}_{name}"] = float(val)
        row_dict[f"{prefix}_{name}_lo"] = float(lo)
        row_dict[f"{prefix}_{name}_hi"] = float(hi)


def format_param_box(title: str, param_names: List[str], params: np.ndarray, ci: np.ndarray, rmse_mV: float, mae_mV: float, r2: float, sigma2: float, sigma2_ci: Tuple[float, float]) -> str:
    lines = [title]
    for name, val, (lo, hi) in zip(param_names, params, ci):
        lines.append(f"{name} = {val:.4g}  [{lo:.4g}, {hi:.4g}]")
    lines.append(f"RMSE = {rmse_mV:.2f} mV")
    lines.append(f"MAE  = {mae_mV:.2f} mV")
    lines.append(f"R²   = {r2:.4f}" if np.isfinite(r2) else "R²   = nan")
    lines.append(f"σ²   = {sigma2:.4g}  [{sigma2_ci[0]:.4g}, {sigma2_ci[1]:.4g}]")
    return "\n".join(lines)


# ------------------------------
# Model and fitting
# ------------------------------
def nernst_based_model(C: np.ndarray, E0: float, k1: float, k2: float, A: float, B: float, R: float, *, Q_full: float, discharge_current: float) -> np.ndarray:
    soc = 1.0 - C / Q_full
    soc = np.clip(soc, 1e-6, 1.0 - 1e-6)
    E = E0 + k1 * np.log(soc) + k2 * np.log(1.0 - soc)
    return E - R * discharge_current + A * np.exp(-B * C)


def fit_least_squares_with_starts(residual_fun, starts: List[np.ndarray], bounds: Tuple[np.ndarray, np.ndarray]):
    best_res = None
    best_start = None
    errors = []
    for x0 in starts:
        try:
            res = least_squares(residual_fun, x0=x0, bounds=bounds, method="trf", loss="linear", max_nfev=25000)
            if best_res is None or res.cost < best_res.cost:
                best_res = res
                best_start = x0
        except Exception as e:
            errors.append(str(e))
    meta: Dict[str, Any] = {"method": "least_squares", "num_starts": len(starts), "start_used": None if best_start is None else best_start.tolist()}
    if errors:
        meta["start_errors"] = errors
    return best_res, meta


def fit_model_ls(C_arr: np.ndarray, V_target: np.ndarray, Q_full: float, discharge_current: float) -> Tuple[np.ndarray, Dict[str, Any], Tuple[np.ndarray, np.ndarray]]:
    lower = np.array([0.5, -3.0, -3.0, -3.0, 1e-4, 0.0], dtype=float)
    upper = np.array([4.5,  3.0,  3.0,  3.0, 50.0, 5.0], dtype=float)
    bounds = (lower, upper)

    y0 = float(np.nanmedian(V_target)) if np.isfinite(np.nanmedian(V_target)) else 3.0
    x0 = np.array([y0, -0.1, 0.1, 0.0, 1.0, 0.1], dtype=float)
    starts = [
        x0,
        np.array([y0, -0.2, 0.2, 0.1, 0.5, 0.05], dtype=float),
        np.array([y0, -0.05, 0.05, 0.0, 2.0, 0.2], dtype=float),
    ]

    def residuals(p):
        pred = nernst_based_model(C_arr, p[0], p[1], p[2], p[3], p[4], p[5], Q_full=Q_full, discharge_current=discharge_current)
        return pred - V_target

    res, meta = fit_least_squares_with_starts(residuals, starts, bounds=bounds)
    if res is None:
        meta["success"] = False
        meta["message"] = "All least_squares starts failed"
        return x0, meta, bounds

    p_hat = res.x
    n = len(V_target)
    p = len(p_hat)
    dof = max(n - p, 1)
    resid = res.fun
    rss = float(np.sum(resid ** 2))
    s2 = rss / dof
    try:
        cov = s2 * np.linalg.pinv(res.jac.T @ res.jac)
    except Exception:
        cov = np.full((p, p), np.nan)

    meta.update({
        "success": bool(res.success),
        "message": res.message,
        "rss": rss,
        "sigma2": s2,
        "dof": dof,
        "pcov": cov,
        "residuals": resid,
        "cost": float(res.cost),
        "nfev": int(getattr(res, "nfev", -1)),
    })
    return p_hat, meta, bounds


def rss_for_params(C_arr: np.ndarray, V_target: np.ndarray, params: np.ndarray, Q_full: float, discharge_current: float) -> float:
    pred = nernst_based_model(C_arr, *params, Q_full=Q_full, discharge_current=discharge_current)
    resid = pred - V_target
    resid = resid[np.isfinite(resid)]
    return float(np.sum(resid ** 2))


def insert_fixed_params(fixed_idx: int, fixed_value: float, free_values: np.ndarray, template: np.ndarray) -> np.ndarray:
    p = np.array(template, dtype=float).copy()
    free_positions = [k for k in range(len(p)) if k != fixed_idx]
    p[fixed_idx] = fixed_value
    p[free_positions] = free_values
    return p


# ------------------------------
# Bootstrap / profile likelihood
# ------------------------------
def profile_likelihood_one(C_arr: np.ndarray, V_target: np.ndarray, params_hat: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray], Q_full: float, discharge_current: float, idx: int, *, alpha: float = 0.05, grid_size: int = 55, span_frac: float = 0.30):
    lower, upper = bounds
    p0 = float(params_hat[idx])
    span = span_frac * max(abs(p0), 1.0)
    lo = max(lower[idx], p0 - span)
    hi = min(upper[idx], p0 + span)
    grid = np.linspace(lo, hi, grid_size)
    free_idx = [k for k in range(len(params_hat)) if k != idx]
    free0 = params_hat[free_idx]
    free_bounds = (lower[free_idx], upper[free_idx])

    profile_rss = np.full_like(grid, np.nan, dtype=float)
    for g, value in enumerate(grid):
        if not np.isfinite(value):
            continue

        def residuals_free(free):
            p = insert_fixed_params(idx, value, free, params_hat)
            return nernst_based_model(C_arr, *p, Q_full=Q_full, discharge_current=discharge_current) - V_target

        try:
            res = least_squares(residuals_free, x0=free0, bounds=free_bounds, method="trf", loss="linear", max_nfev=15000)
            profile_rss[g] = float(np.sum(res.fun ** 2))
        except Exception:
            profile_rss[g] = np.nan

    rss_min = float(np.nanmin(profile_rss))
    sigma2 = float(np.sum((nernst_based_model(C_arr, *params_hat, Q_full=Q_full, discharge_current=discharge_current) - V_target) ** 2) / max(len(V_target) - len(params_hat), 1))
    delta_thr = sigma2 * float(chi2.ppf(1.0 - alpha, 1))
    return grid, profile_rss, rss_min, delta_thr


def profile_ci_from_curve(grid: np.ndarray, profile_rss: np.ndarray, rss_min: float, delta_thr: float) -> Tuple[float, float]:
    y = profile_rss - rss_min - delta_thr
    ok = np.isfinite(y)
    grid = grid[ok]
    y = y[ok]
    if len(grid) < 3 or np.all(y > 0):
        return np.nan, np.nan
    idx_min = int(np.nanargmin(profile_rss))

    def interp_cross(left: bool) -> float:
        if left:
            for k in range(idx_min, 0, -1):
                if y[k] <= 0 and y[k - 1] > 0:
                    x1, x2 = grid[k - 1], grid[k]
                    y1, y2 = y[k - 1], y[k]
                    return float(x1 + (0 - y1) * (x2 - x1) / (y2 - y1))
        else:
            for k in range(idx_min, len(grid) - 1):
                if y[k] <= 0 and y[k + 1] > 0:
                    x1, x2 = grid[k], grid[k + 1]
                    y1, y2 = y[k], y[k + 1]
                    return float(x1 + (0 - y1) * (x2 - x1) / (y2 - y1))
        return np.nan

    return interp_cross(True), interp_cross(False)


def bootstrap_params_residual(C_arr: np.ndarray, V_target: np.ndarray, params_hat: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray], Q_full: float, discharge_current: float, *, n_boot: int = 200, random_seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    y_hat = nernst_based_model(C_arr, *params_hat, Q_full=Q_full, discharge_current=discharge_current)
    resid = V_target - y_hat
    resid = resid[np.isfinite(resid)]
    if len(resid) < 3:
        return np.empty((0, len(params_hat)), dtype=float)

    samples = []
    for _ in range(n_boot):
        boot_resid = rng.choice(resid, size=len(resid), replace=True)
        y_star = y_hat + boot_resid

        def residuals(p):
            return nernst_based_model(C_arr, *p, Q_full=Q_full, discharge_current=discharge_current) - y_star

        try:
            res = least_squares(residuals, x0=params_hat, bounds=bounds, method="trf", loss="linear", max_nfev=15000)
            if res.success:
                samples.append(res.x)
        except Exception:
            pass
    return np.array(samples, dtype=float)


def choose_most_correlated_pair(samples: Optional[np.ndarray], fallback_cov: np.ndarray) -> Tuple[int, int, float]:
    if samples is not None and len(samples) >= 3:
        corr = np.corrcoef(samples, rowvar=False)
        corr = np.where(np.isfinite(corr), corr, 0.0)
        np.fill_diagonal(corr, 0.0)
        i, j = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
        return int(i), int(j), float(corr[i, j])
    cov = np.asarray(fallback_cov, dtype=float)
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, np.inf))
    corr = cov / np.outer(d, d)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    np.fill_diagonal(corr, 0.0)
    i, j = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    return int(i), int(j), float(corr[i, j])


# ------------------------------
# Plots
# ------------------------------
def plot_main_fit(C_arr: np.ndarray, V_target: np.ndarray, params_hat: np.ndarray, Q_full: float, discharge_current: float, out_path: Path, title: str, *, ci: Optional[np.ndarray] = None, sigma2: Optional[float] = None, sigma2_ci: Optional[Tuple[float, float]] = None, rmse_mV: Optional[float] = None, mae_mV: Optional[float] = None, r2: Optional[float] = None):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(C_arr, V_target, linewidth=0.8, alpha=0.35, label="Data")
    C_grid = np.linspace(0, max(np.max(C_arr), Q_full * 0.995), 400)
    y_fit = nernst_based_model(C_grid, *params_hat, Q_full=Q_full, discharge_current=discharge_current)
    ax.plot(C_grid, y_fit, linewidth=2.0, label="Fit")
    ax.set_xlabel("Consumed capacity C (Ah)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    if ci is not None and sigma2 is not None and sigma2_ci is not None and rmse_mV is not None and mae_mV is not None and r2 is not None:
        box = format_param_box("Nernst", ["E0", "k1", "k2", "A", "B", "R"], params_hat, ci, rmse_mV, mae_mV, r2, sigma2, sigma2_ci)
        ax.text(0.01, 0.01, box, transform=ax.transAxes, va="bottom", ha="left", fontsize=8.1, family="monospace", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"))
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pair_surfaces(C_arr: np.ndarray, V_target: np.ndarray, params_hat: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray], Q_full: float, discharge_current: float, out_dir: Path, param_names: List[str], *, grid_size: int = 60, span_frac: float = 0.30, levels: int = 30, bootstrap_samples: Optional[np.ndarray] = None, surface_pdf_name: str = "all_error_surfaces.pdf"):
    out_dir.mkdir(parents=True, exist_ok=True)
    lower, upper = bounds
    pdf_path = out_dir / surface_pdf_name
    with PdfPages(pdf_path) as pdf:
        for i in range(len(params_hat)):
            for j in range(i + 1, len(params_hat)):
                pi0 = float(params_hat[i])
                pj0 = float(params_hat[j])
                di = span_frac * max(abs(pi0), 1.0)
                dj = span_frac * max(abs(pj0), 1.0)
                lo_i = max(lower[i], pi0 - di)
                hi_i = min(upper[i], pi0 + di)
                lo_j = max(lower[j], pj0 - dj)
                hi_j = min(upper[j], pj0 + dj)
                xi = np.linspace(lo_i, hi_i, grid_size)
                xj = np.linspace(lo_j, hi_j, grid_size)
                X, Y = np.meshgrid(xi, xj)
                Z = np.full_like(X, np.nan, dtype=float)
                base = np.array(params_hat, dtype=float)
                for r in range(grid_size):
                    for c in range(grid_size):
                        trial = base.copy()
                        trial[i] = X[r, c]
                        trial[j] = Y[r, c]
                        try:
                            Z[r, c] = rss_for_params(C_arr, V_target, trial, Q_full, discharge_current)
                        except Exception:
                            pass
                if np.all(~np.isfinite(Z)):
                    continue
                Zp = np.log10(np.clip(Z, 1e-18, np.inf))
                fig, ax = plt.subplots(figsize=(8.6, 6.7))
                cf = ax.contourf(X, Y, Zp, levels=levels)
                ax.contour(X, Y, Zp, levels=levels, linewidths=0.4, colors="black", alpha=0.25)
                ax.plot(pi0, pj0, marker="x", markersize=10, mew=2, color="red")
                if bootstrap_samples is not None and len(bootstrap_samples) > 0:
                    ax.scatter(bootstrap_samples[:, i], bootstrap_samples[:, j], s=10, alpha=0.20, color="white", edgecolors="none")
                ax.set_xlabel(param_names[i])
                ax.set_ylabel(param_names[j])
                ax.set_title(f"RSS surface: {param_names[i]} vs {param_names[j]}")
                cbar = fig.colorbar(cf, ax=ax)
                cbar.set_label("log10(RSS)")
                ax.grid(True, linestyle="--", alpha=0.35)
                plt.tight_layout()
                png_name = f"surface_{param_names[i]}__{param_names[j]}.png"
                fig.savefig(out_dir / png_name, dpi=160)
                pdf.savefig(fig)
                plt.close(fig)


def plot_profile_likelihoods(C_arr: np.ndarray, V_target: np.ndarray, params_hat: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray], Q_full: float, discharge_current: float, out_dir: Path, param_names: List[str], *, alpha: float = 0.05, grid_size: int = 55, span_frac: float = 0.30) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_params = len(params_hat)
    ncols = 3
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), constrained_layout=True)
    axes = np.ravel(np.array(axes))
    for idx in range(n_params):
        grid, prof_rss, rss_min, delta_thr = profile_likelihood_one(C_arr, V_target, params_hat, bounds, Q_full, discharge_current, idx, alpha=alpha, grid_size=grid_size, span_frac=span_frac)
        ci_lo, ci_hi = profile_ci_from_curve(grid, prof_rss, rss_min, delta_thr)
        rows.append({"param": param_names[idx], "estimate": float(params_hat[idx]), "ci_low": ci_lo, "ci_high": ci_hi, "profile_rss_min": rss_min, "delta_threshold": delta_thr})
        ax = axes[idx]
        ax.plot(grid, prof_rss, linewidth=1.8)
        ax.axhline(rss_min + delta_thr, linestyle="--", linewidth=1.2)
        ax.axvline(params_hat[idx], linestyle=":", linewidth=1.2)
        if np.isfinite(ci_lo):
            ax.axvspan(ci_lo, ci_hi, alpha=0.18)
        ax.set_title(param_names[idx])
        ax.set_xlabel(param_names[idx])
        ax.set_ylabel("Profile RSS")
        ax.grid(True, linestyle="--", alpha=0.3)
    for k in range(n_params, len(axes)):
        axes[k].axis("off")
    pdf_path = out_dir / "profile_likelihood.pdf"
    png_path = out_dir / "profile_likelihood.png"
    fig.suptitle("Profile likelihood curves", y=1.02, fontsize=14)
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_equivalence_region(C_arr: np.ndarray, V_target: np.ndarray, params_hat: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray], Q_full: float, discharge_current: float, bootstrap_samples: Optional[np.ndarray], out_path: Path, param_names: List[str], *, alpha: float = 0.05, grid_size: int = 80, span_frac: float = 0.35):
    if bootstrap_samples is not None and len(bootstrap_samples) >= 3:
        i, j, corr = choose_most_correlated_pair(bootstrap_samples, np.eye(len(params_hat)))
    else:
        i, j, corr = choose_most_correlated_pair(None, np.asarray(np.eye(len(params_hat))))

    lower, upper = bounds
    pi0 = float(params_hat[i])
    pj0 = float(params_hat[j])
    di = span_frac * max(abs(pi0), 1.0)
    dj = span_frac * max(abs(pj0), 1.0)
    lo_i = max(lower[i], pi0 - di)
    hi_i = min(upper[i], pi0 + di)
    lo_j = max(lower[j], pj0 - dj)
    hi_j = min(upper[j], pj0 + dj)
    xi = np.linspace(lo_i, hi_i, grid_size)
    xj = np.linspace(lo_j, hi_j, grid_size)
    X, Y = np.meshgrid(xi, xj)
    Z = np.full_like(X, np.nan, dtype=float)
    base = np.array(params_hat, dtype=float)
    rss0 = rss_for_params(C_arr, V_target, base, Q_full, discharge_current)
    for r in range(grid_size):
        for c in range(grid_size):
            trial = base.copy()
            trial[i] = X[r, c]
            trial[j] = Y[r, c]
            try:
                Z[r, c] = rss_for_params(C_arr, V_target, trial, Q_full, discharge_current) - rss0
            except Exception:
                pass
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    Zp = np.log10(np.clip(Z + 1e-18, 1e-18, np.inf))
    cf = ax.contourf(X, Y, Zp, levels=30)
    ax.contour(X, Y, Z, levels=[np.nanpercentile(Z[np.isfinite(Z)], 10) if np.any(np.isfinite(Z)) else 0.0], colors="white", linewidths=2.2)
    ax.contour(X, Y, Z, levels=np.linspace(np.nanmin(Z[np.isfinite(Z)]) if np.any(np.isfinite(Z)) else 0.0, np.nanmax(Z[np.isfinite(Z)]) if np.any(np.isfinite(Z)) else 1.0, 8), colors="black", alpha=0.18, linewidths=0.7)
    ax.plot(pi0, pj0, marker="x", markersize=12, mew=2.5, color="red")
    if bootstrap_samples is not None and len(bootstrap_samples) > 0:
        ax.scatter(bootstrap_samples[:, i], bootstrap_samples[:, j], s=18, alpha=0.22, color="cyan", edgecolors="none", label="Bootstrap samples")
        ax.legend(loc="best")
    ax.set_xlabel(param_names[i])
    ax.set_ylabel(param_names[j])
    ax.set_title(f"Near-equivalent solution region: {param_names[i]} vs {param_names[j]} (corr={corr:.3f})")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("log10(ΔRSS)")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {"pair_i": param_names[i], "pair_j": param_names[j], "corr": float(corr)}


def save_bootstrap_pairplot(bootstrap_samples: np.ndarray, out_path: Path, param_names: List[str]):
    if bootstrap_samples is None or len(bootstrap_samples) == 0:
        return
    i, j, corr = choose_most_correlated_pair(bootstrap_samples, np.eye(bootstrap_samples.shape[1]))
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.scatter(bootstrap_samples[:, i], bootstrap_samples[:, j], s=18, alpha=0.3)
    ax.set_xlabel(param_names[i])
    ax.set_ylabel(param_names[j])
    ax.set_title(f"Bootstrap cloud: {param_names[i]} vs {param_names[j]} (corr={corr:.3f})")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# ------------------------------
# Main record processing
# ------------------------------
def fit_and_analyze_one(row: pd.Series, out_dir: Path, *, time_unit: str = "minutes", voltage_key: Optional[str] = None, alpha: float = 0.05, bad_rmse_mv: float = 50.0, bad_r2: float = 0.95, bad_subdir: str = "bad_fit", save_surfaces: bool = True, save_bootstrap: bool = True, save_profile: bool = True, bootstrap_n: int = 200, bootstrap_seed: int = 42, surface_grid_size: int = 60, surface_span_frac: float = 0.30) -> Dict[str, Any]:
    test_id = row.get("test_id", "")
    brand_model = str(row.get("brand_model", ""))
    element_type = infer_element_type(row)

    info = parse_jsonish(row.get("info"), default={})
    measurements = parse_jsonish(row.get("measurements"), default=[])
    discharge_curve = parse_jsonish(row.get("discharge_curve"), default={})

    vk, subk, time_list, volt_list = choose_voltage_series(discharge_curve, preferred_key=voltage_key)
    time_arr = safe_float_array(time_list)
    volt_arr = safe_float_array(volt_list)
    mask = np.isfinite(time_arr) & np.isfinite(volt_arr)
    time_arr = time_arr[mask]
    volt_arr = volt_arr[mask]
    if len(time_arr) < 5:
        raise ValueError("Слишком мало валидных точек для фита")

    if time_unit == "minutes":
        time_hours = time_arr / 60.0
    elif time_unit == "seconds":
        time_hours = time_arr / 3600.0
    elif time_unit == "hours":
        time_hours = time_arr.copy()
    else:
        raise ValueError("time_unit must be minutes/seconds/hours")

    time_hours = time_hours - time_hours[0]
    total_time_h = float(time_hours[-1])
    if not np.isfinite(total_time_h) or total_time_h <= 0:
        raise ValueError("Некорректная длительность разряда")

    cap_mah = extract_capacity_mAh(measurements, info=info)
    if cap_mah is None or not np.isfinite(cap_mah):
        cap_mah = 1000.0
    Q_full = float(cap_mah) / 1000.0
    discharge_current = Q_full / total_time_h
    C_arr = discharge_current * time_hours
    _ = smooth_voltage(volt_arr)  # kept for consistency / possible future use

    params_hat, meta, bounds = fit_model_ls(C_arr, volt_arr, Q_full, discharge_current)
    V_fit = nernst_based_model(C_arr, *params_hat, Q_full=Q_full, discharge_current=discharge_current)
    rmse = float(np.sqrt(np.mean((volt_arr - V_fit) ** 2)))
    mae = float(np.mean(np.abs(volt_arr - V_fit)))
    r2 = float(r2_score(volt_arr, V_fit))
    ci = param_confint(params_hat, meta["pcov"], meta["dof"], alpha=alpha)
    sigma2, sigma2_ci = variance_confint_from_residuals(meta["residuals"], p=len(params_hat), alpha=alpha)
    is_bad_fit = (rmse * 1000.0 > bad_rmse_mv) or (r2 < bad_r2)

    base_dir = (out_dir / bad_subdir) if is_bad_fit else out_dir
    type_dir = base_dir / element_type
    type_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{int(test_id) if str(test_id).isdigit() else test_id}_{sanitize_filename(brand_model)}"

    fit_png = type_dir / f"{stem}_fit.png"
    plot_main_fit(C_arr, volt_arr, params_hat, Q_full, discharge_current, fit_png, f"{element_type.title()} | {brand_model} | test_id={test_id}", ci=ci, sigma2=sigma2, sigma2_ci=sigma2_ci, rmse_mV=rmse * 1000.0, mae_mV=mae * 1000.0, r2=r2)

    result = {
        "test_id": test_id,
        "brand_model": brand_model,
        "element_type": element_type,
        "voltage_key": vk,
        "voltage_subkey": subk,
        "capacity_mAh": float(cap_mah),
        "Q_full_Ah": float(Q_full),
        "total_time_h": total_time_h,
        "discharge_current_A": float(discharge_current),
        "num_points": int(len(C_arr)),
        "rmse_mV": rmse * 1000.0,
        "mae_mV": mae * 1000.0,
        "r2": r2,
        "sigma2": float(sigma2),
        "sigma2_lo": float(sigma2_ci[0]),
        "sigma2_hi": float(sigma2_ci[1]),
        "plot_path": str(fit_png),
        "status": "bad_fit" if is_bad_fit else "ok",
        "is_bad_fit": bool(is_bad_fit),
        "bad_rmse_mv_threshold": float(bad_rmse_mv),
        "bad_r2_threshold": float(bad_r2),
        "fit_method": meta.get("method", ""),
    }
    add_param_columns(result, "nernst", ["E0", "k1", "k2", "A", "B", "R"], params_hat, ci)

    diagnostics_dir = type_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_samples = None
    if save_bootstrap:
        bootstrap_samples = bootstrap_params_residual(C_arr, volt_arr, params_hat, bounds, Q_full, discharge_current, n_boot=bootstrap_n, random_seed=bootstrap_seed)
        if len(bootstrap_samples) > 0:
            bdf = pd.DataFrame(bootstrap_samples, columns=["E0", "k1", "k2", "A", "B", "R"])
            bdf.to_csv(diagnostics_dir / "bootstrap_samples.csv", index=False)
            save_bootstrap_pairplot(bootstrap_samples, diagnostics_dir / "bootstrap_cloud.png", ["E0", "k1", "k2", "A", "B", "R"])
            result["bootstrap_n_success"] = int(len(bootstrap_samples))
            for k, pname in enumerate(["E0", "k1", "k2", "A", "B", "R"]):
                result[f"bootstrap_{pname}_mean"] = float(np.mean(bootstrap_samples[:, k]))
                result[f"bootstrap_{pname}_std"] = float(np.std(bootstrap_samples[:, k], ddof=1))
        else:
            result["bootstrap_n_success"] = 0

    if save_surfaces:
        surf_dir = diagnostics_dir / "error_surfaces"
        plot_pair_surfaces(C_arr, volt_arr, params_hat, bounds, Q_full, discharge_current, surf_dir, ["E0", "k1", "k2", "A", "B", "R"], grid_size=surface_grid_size, span_frac=surface_span_frac, bootstrap_samples=bootstrap_samples)

    if save_profile:
        profile_dir = diagnostics_dir / "profile_likelihood"
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_df = plot_profile_likelihoods(C_arr, volt_arr, params_hat, bounds, Q_full, discharge_current, profile_dir, ["E0", "k1", "k2", "A", "B", "R"], alpha=alpha)
        profile_df.to_csv(profile_dir / "profile_ci.csv", index=False)
        result["profile_ci_path"] = str(profile_dir / "profile_ci.csv")

    eq_dir = diagnostics_dir / "equivalence_region"
    eq_dir.mkdir(parents=True, exist_ok=True)
    eq_meta = plot_equivalence_region(C_arr, volt_arr, params_hat, bounds, Q_full, discharge_current, bootstrap_samples, eq_dir / "equivalence_region.png", ["E0", "k1", "k2", "A", "B", "R"], alpha=alpha)
    pd.DataFrame([eq_meta]).to_csv(eq_dir / "equivalence_region_meta.csv", index=False)
    result.update({f"equiv_{k}": v for k, v in eq_meta.items()})

    return result


# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Contour plots, bootstrap and profile likelihood for one battery discharge curve.")
    parser.add_argument("input_csv", help="Path to CSV file")
    parser.add_argument("output_dir", nargs="?", default="fit_check", help="Output directory")
    parser.add_argument("--time-unit", default="minutes", choices=["minutes", "seconds", "hours"], help="Units of time in discharge_curve")
    parser.add_argument("--voltage-key", default=None, help="Preferred curve key")
    parser.add_argument("--limit", type=int, default=1, help="How many rows to process (default: 1)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--bad-rmse-mv", type=float, default=50.0, help="Bad fit threshold by RMSE in mV")
    parser.add_argument("--bad-r2", type=float, default=0.95, help="Bad fit threshold by R²")
    parser.add_argument("--bad-subdir", default="bad_fit", help="Subdirectory for bad fits")
    parser.add_argument("--bootstrap-n", type=int, default=200, help="Bootstrap replicates")
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="Bootstrap random seed")
    parser.add_argument("--no-bootstrap", action="store_true", help="Disable bootstrap")
    parser.add_argument("--no-profile", action="store_true", help="Disable profile likelihood")
    parser.add_argument("--no-surfaces", action="store_true", help="Disable pairwise RSS surfaces")
    parser.add_argument("--surface-grid-size", type=int, default=60, help="Grid size for pairwise surfaces")
    parser.add_argument("--surface-span-frac", type=float, default=0.30, help="Span fraction for surfaces")
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df = df.head(args.limit) if args.limit is not None else df.head(1)

    print(f"Rows in dataset: {len(df)}")
    print(f"Output directory: {out_dir.resolve()}")

    results = []
    errors = []
    for idx, row in df.iterrows():
        try:
            res = fit_and_analyze_one(
                row,
                out_dir,
                time_unit=args.time_unit,
                voltage_key=args.voltage_key,
                alpha=args.alpha,
                bad_rmse_mv=args.bad_rmse_mv,
                bad_r2=args.bad_r2,
                bad_subdir=args.bad_subdir,
                save_surfaces=not args.no_surfaces,
                save_bootstrap=not args.no_bootstrap,
                save_profile=not args.no_profile,
                bootstrap_n=args.bootstrap_n,
                bootstrap_seed=args.bootstrap_seed,
                surface_grid_size=args.surface_grid_size,
                surface_span_frac=args.surface_span_frac,
            )
            results.append(res)
            print(f"[{idx + 1}/{len(df)}] OK test_id={res['test_id']} {res['brand_model']}")
        except Exception as e:
            errors.append({"test_id": row.get("test_id", ""), "brand_model": row.get("brand_model", ""), "error": str(e)})
            print(f"[{idx + 1}/{len(df)}] ERR test_id={row.get('test_id', '')} {row.get('brand_model', '')} -> {e}")

    results_df = pd.DataFrame(results)
    errors_df = pd.DataFrame(errors)
    results_df.to_csv(out_dir / "fit_summary.csv", index=False)
    if len(errors_df) > 0:
        errors_df.to_csv(out_dir / "fit_errors.csv", index=False)

    print("\nDone.")
    print(f"Successful fits: {len(results_df)}")
    print(f"Errors: {len(errors_df)}")
    print(f"Summary: {out_dir / 'fit_summary.csv'}")
    if len(errors_df) > 0:
        print(f"Errors: {out_dir / 'fit_errors.csv'}")


if __name__ == "__main__":
    main()
