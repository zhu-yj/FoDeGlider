from __future__ import annotations

import argparse
import copy
import csv
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(THIS_DIR)
BASE_DIR = CODE_DIR
PROJECT_ROOT_STR = os.path.dirname(CODE_DIR)
IDENT_DIR = os.path.join(CODE_DIR, "identification")
os.environ.setdefault("SOURCE_PROJ_DIR", PROJECT_ROOT_STR)
for _path in (THIS_DIR, IDENT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from SI_awug_crba_utils import (  # noqa: E402
    AWUGModelFromTex,
    apply_body_state,
    apply_gate_state,
    apply_wing_gd_state,
    build_traj_data,
    clone_base_params,
    get_integrator_step,
    load_norm_scale,
    load_stage_json,
    parse_files,
    save_json,
    start_stage,
)
from SI_awug_e import PositiveQuadraticBernsteinModel  # noqa: E402


DEFAULT_STAGE_E_JSON = os.path.join(
    BASE_DIR,
    "log",
    "si_awug_crba_final_wing",
    "stage_e",
    "20260603_103531",
    "stage_e_result.json",
)

STATE_CHANNEL_LABELS = [
    "p_x",
    "p_y",
    "p_z",
    "e_phi",
    "e_theta",
    "e_psi",
    "v_x",
    "v_y",
    "v_z",
    "w_x",
    "w_y",
    "w_z",
]

GROUP_SLICES = {
    "p": slice(0, 3),
    "e": slice(3, 6),
    "v": slice(6, 9),
    "w": slice(9, 12),
}

VARIANT_SPECS: List[Tuple[str, Dict[str, bool]]] = [
    (
        "full_model",
        {
            "disable_force_gate": False,
            "disable_torque_gate": False,
            "disable_added_mass_inertia_gates": False,
            "disable_added_mass_gates": False,
            "disable_added_inertia_gates": False,
            "linear_added_mass_gates": False,
            "linear_added_inertia_gates": False,
        },
    ),
    (
        "no_force_gate",
        {
            "disable_force_gate": True,
            "disable_torque_gate": False,
            "disable_added_mass_inertia_gates": False,
            "disable_added_mass_gates": False,
            "disable_added_inertia_gates": False,
            "linear_added_mass_gates": False,
            "linear_added_inertia_gates": False,
        },
    ),
    (
        "no_torque_gate",
        {
            "disable_force_gate": False,
            "disable_torque_gate": True,
            "disable_added_mass_inertia_gates": False,
            "disable_added_mass_gates": False,
            "disable_added_inertia_gates": False,
            "linear_added_mass_gates": False,
            "linear_added_inertia_gates": False,
        },
    ),
    (
        "no_added_mass_gate",
        {
            "disable_force_gate": False,
            "disable_torque_gate": False,
            "disable_added_mass_inertia_gates": False,
            "disable_added_mass_gates": True,
            "disable_added_inertia_gates": False,
            "linear_added_mass_gates": False,
            "linear_added_inertia_gates": False,
        },
    ),
    (
        "no_added_inertia_gate",
        {
            "disable_force_gate": False,
            "disable_torque_gate": False,
            "disable_added_mass_inertia_gates": False,
            "disable_added_mass_gates": False,
            "disable_added_inertia_gates": True,
            "linear_added_mass_gates": False,
            "linear_added_inertia_gates": False,
        },
    ),
]


class ForceTorqueGateAblationMixin:
    """Model variant that can turn off X5 and X6 sweep gates independently."""

    def __init__(
        self,
        *args,
        disable_force_gate: bool = False,
        disable_torque_gate: bool = False,
        disable_added_mass_gate: bool = False,
        disable_added_inertia_gate: bool = False,
        linear_added_mass_gate: bool = False,
        linear_added_inertia_gate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.disable_force_gate = bool(disable_force_gate)
        self.disable_torque_gate = bool(disable_torque_gate)
        self.disable_added_mass_gate = bool(disable_added_mass_gate)
        self.disable_added_inertia_gate = bool(disable_added_inertia_gate)
        self.linear_added_mass_gate = bool(linear_added_mass_gate)
        self.linear_added_inertia_gate = bool(linear_added_inertia_gate)

    def _construct_X5_matrix(self, *args, **kwargs):
        old = self.enable_sweep_gates
        if self.disable_force_gate:
            self.enable_sweep_gates = False
        try:
            return super()._construct_X5_matrix(*args, **kwargs)
        finally:
            self.enable_sweep_gates = old

    def _construct_X6_matrices(self, *args, **kwargs):
        old = self.enable_sweep_gates
        if self.disable_torque_gate:
            self.enable_sweep_gates = False
        try:
            return super()._construct_X6_matrices(*args, **kwargs)
        finally:
            self.enable_sweep_gates = old

    def _wing_added_mass_gates(self, theta_k: torch.Tensor):
        g_ma, g_ia = super()._wing_added_mass_gates(theta_k)
        theta_abs = torch.abs(theta_k)
        eta_s, _, eta_kn = self._gate_eta_terms(theta_abs)
        if self.linear_added_mass_gate:
            g_ma = torch.stack([eta_s, eta_s, eta_s]).reshape(3)
        if self.linear_added_inertia_gate:
            g_ia = torch.stack([eta_kn, eta_s, eta_kn]).reshape(3)
        if self.disable_added_mass_gate:
            g_ma = torch.ones_like(g_ma)
        if self.disable_added_inertia_gate:
            g_ia = torch.ones_like(g_ia)
        return g_ma, g_ia


class ForceTorqueCategoricalAblationModel(ForceTorqueGateAblationMixin, AWUGModelFromTex):
    """Categorical Stage E force/torque gate ablation model."""


class ForceTorqueBernsteinAblationModel(ForceTorqueGateAblationMixin, PositiveQuadraticBernsteinModel):
    """Positive-quadratic Bernstein Stage E force/torque gate ablation model."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Stage E force/torque and added-mass/inertia sweep-gate ablations. "
            "Force gate = X5/Theta_5 gates; torque gate = X6/Theta_6 gates; "
            "added-mass/inertia gates scale wing added mass and inertia."
        )
    )
    parser.add_argument(
        "--input-e-json",
        default=DEFAULT_STAGE_E_JSON,
        help="Path to stage_e_result.json. Defaults to the 20260603_103531 Stage E result.",
    )
    parser.add_argument("--input-b-json", default=None, help="Override Stage B JSON path.")
    parser.add_argument("--input-d-json", default=None, help="Override Stage D JSON path.")
    parser.add_argument("--test-files", default=None, help="Comma-separated held-out trajectory files.")
    parser.add_argument("--out-dir", default="log/si_awug_crba_final_wing")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-step", type=int, default=None)
    parser.add_argument("--dt-base", type=float, default=None)
    parser.add_argument(
        "--integrator",
        choices=["rk2", "rk4"],
        default=None,
        help="Rollout integrator. Defaults to meta.integrator from Stage E.",
    )
    parser.add_argument(
        "--horizons",
        default=None,
        help="Comma-separated rollout horizons. Defaults to Stage E window_batch_size, then 360.",
    )
    parser.add_argument(
        "--batch-skip",
        type=int,
        default=0,
        help="Stride between rollout windows. Use 0 to set batch_skip=horizon for each horizon.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variants, or all. Available: " + ",".join(name for name, _ in VARIANT_SPECS),
    )
    parser.add_argument(
        "--norm-mode",
        choices=["auto", "none", "minmax", "zscore"],
        default="auto",
        help="Use auto to inherit Stage E meta.norm_mode.",
    )
    parser.add_argument("--norm-stats-json", type=str, default=None)
    parser.add_argument(
        "--min-sweep-deg",
        type=float,
        default=0.0,
        help="Only evaluate windows whose max absolute left/right sweep angle reaches this threshold.",
    )
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=0,
        help="Optional smoke-test limit per file. Use 0 for all windows.",
    )
    return parser


def _parse_int_list(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one horizon")
    if any(v <= 1 for v in vals):
        raise ValueError("all horizon values must be > 1")
    return vals


def _select_variant_specs(raw: str) -> List[Tuple[str, Dict[str, bool]]]:
    if not raw or raw.strip().lower() == "all":
        return VARIANT_SPECS

    available = {name: flags for name, flags in VARIANT_SPECS}
    selected: List[Tuple[str, Dict[str, bool]]] = []
    unknown: List[str] = []
    for name in [x.strip() for x in str(raw).split(",") if x.strip()]:
        if name not in available:
            unknown.append(name)
        else:
            selected.append((name, available[name]))
    if unknown:
        raise ValueError(f"unknown variants {unknown}; available={list(available)} or all")
    if not selected:
        raise ValueError("expected at least one variant")
    return selected


def _resolve_optional_path(path: Optional[str], *, base_dir: str) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return _relocate_workspace_path(path) or path
    candidate = os.path.join(base_dir, path)
    return _relocate_workspace_path(candidate) or candidate


def _relocate_workspace_path(path: Optional[str]) -> Optional[str]:
    """Map stale absolute paths from another checkout to this workspace."""
    if not path:
        return None
    text = str(path)
    if os.path.exists(text):
        return text

    normalized = text.replace("\\", "/")
    marker = "stage_a_to_e_abnode_style/"
    if marker in normalized:
        tail = normalized.split(marker, 1)[1]
        candidate = os.path.join(BASE_DIR, *tail.split("/"))
        if os.path.exists(candidate):
            return candidate

    code_marker = "code_folder/"
    if code_marker in normalized:
        tail = normalized.split(code_marker, 1)[1]
        candidate = os.path.join(os.path.dirname(BASE_DIR), *tail.split("/"))
        if os.path.exists(candidate):
            return candidate

    basename = os.path.basename(normalized)
    if basename:
        candidate = os.path.join(os.path.dirname(BASE_DIR), basename)
        if os.path.exists(candidate):
            return candidate

    return None


def _meta_list_or_csv(meta: Dict, key: str) -> Optional[str]:
    value = meta.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join(str(x) for x in value)
    return str(value)


def _resolve_config(args, stage_e: Dict) -> Dict[str, object]:
    meta = stage_e.get("meta", {})

    input_b_json = _relocate_workspace_path(args.input_b_json or meta.get("input_b_json")) or (
        args.input_b_json or meta.get("input_b_json")
    )
    input_d_json = _relocate_workspace_path(args.input_d_json or meta.get("input_d_json")) or (
        args.input_d_json or meta.get("input_d_json")
    )
    if not input_b_json:
        raise ValueError("missing Stage B JSON; pass --input-b-json or use Stage E meta.input_b_json")
    if not input_d_json:
        raise ValueError("missing Stage D JSON; pass --input-d-json or use Stage E meta.input_d_json")

    test_files_arg = args.test_files or _meta_list_or_csv(meta, "test_files")
    if not test_files_arg:
        raise ValueError("missing test files; pass --test-files or use Stage E meta.test_files")

    sample_step = int(args.sample_step if args.sample_step is not None else meta.get("sample_step", 1))
    dt_base = float(args.dt_base if args.dt_base is not None else meta.get("dt_base", 1.0 / 90.0))
    integrator = str(args.integrator or meta.get("integrator") or "rk4").lower()
    horizons_raw = args.horizons or str(meta.get("window_batch_size") or meta.get("batch_size_max") or 360)
    horizons = _parse_int_list(horizons_raw)

    norm_mode = str(meta.get("norm_mode") or "none") if args.norm_mode == "auto" else str(args.norm_mode)
    norm_stats_json = args.norm_stats_json if args.norm_stats_json is not None else meta.get("norm_stats_json")

    return {
        "input_b_json": str(input_b_json),
        "input_d_json": str(input_d_json),
        "test_files": parse_files(test_files_arg),
        "sample_step": sample_step,
        "dt_base": dt_base,
        "dt": dt_base * max(1, sample_step),
        "integrator": integrator,
        "horizons": horizons,
        "norm_mode": norm_mode,
        "norm_stats_json": norm_stats_json,
    }


def _is_bernstein_stage(stage_e: Dict) -> bool:
    gate_form = str(stage_e.get("meta", {}).get("gate_form", "")).lower()
    return "bernstein" in gate_form or "wing_bernstein_gates" in stage_e


def _build_model_params(stage_b: Dict, stage_d: Dict) -> Dict:
    params = clone_base_params()
    apply_body_state(params, stage_b)
    apply_wing_gd_state(params, stage_d)
    return params


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _bernstein_gate_upper(stage_e: Dict, key: str) -> Optional[float]:
    meta = stage_e.get("meta", {})
    gates = stage_e.get("wing_bernstein_gates", {})
    for source in (meta, gates):
        value = source.get(key)
        if value is not None:
            return _as_float(value)
    for source in (meta, gates):
        value = source.get("gate_upper")
        if value is not None:
            return _as_float(value)
    return None


def _load_bernstein_state(model: PositiveQuadraticBernsteinModel, stage_e: Dict) -> None:
    gates = stage_e.get("wing_bernstein_gates", {})
    if not gates:
        raise ValueError("Bernstein Stage E JSON is missing wing_bernstein_gates")

    controls = gates.get("control_parameters", {})
    with torch.no_grad():
        added = controls.get("added_mass_inertia", {})
        for name in model.ADDED_GATE_NAMES:
            if name in added:
                getattr(model, name).fill_(_as_float(added[name]))

        for param_name, param_dict in (("k_g2_X5", model.k_g2_X5), ("k_g2_X6", model.k_g2_X6)):
            for key, value in controls.get(param_name, {}).items():
                if key in param_dict:
                    param_dict[key].fill_(_as_float(value))

        # Fallback for older Bernstein JSONs without the compact control tree.
        if not controls.get("k_g2_X5"):
            for channel, (beta_0_name, beta_1_name) in model.HYDRO_X5_CHANNELS.items():
                item = gates.get("hydrodynamic_force", {}).get(channel, {})
                if beta_0_name in model.k_g2_X5 and "beta_0" in item:
                    model.k_g2_X5[beta_0_name].fill_(_as_float(item["beta_0"]))
                if beta_1_name in model.k_g2_X5 and "beta_1" in item:
                    model.k_g2_X5[beta_1_name].fill_(_as_float(item["beta_1"]))

        if not controls.get("k_g2_X6"):
            for channel, (beta_0_name, beta_1_name) in model.HYDRO_X6_CHANNELS.items():
                item = gates.get("hydrodynamic_moment", {}).get(channel, {})
                if beta_0_name in model.k_g2_X6 and "beta_0" in item:
                    model.k_g2_X6[beta_0_name].fill_(_as_float(item["beta_0"]))
                if beta_1_name in model.k_g2_X6 and "beta_1" in item:
                    model.k_g2_X6[beta_1_name].fill_(_as_float(item["beta_1"]))

        if not controls.get("added_mass_inertia"):
            for name in model.ADDED_GATE_NAMES:
                item = gates.get("added_mass_inertia", {}).get(name, {})
                if "beta_1" in item:
                    getattr(model, name).fill_(_as_float(item["beta_1"]))

    model.project_controls_()


def _build_variant_model(
    *,
    device: str,
    params: Dict,
    stage_e: Dict,
    disable_force_gate: bool,
    disable_torque_gate: bool,
    disable_added_mass_inertia_gates: bool,
    disable_added_mass_gates: bool,
    disable_added_inertia_gates: bool,
    linear_added_mass_gates: bool,
    linear_added_inertia_gates: bool,
):
    disable_added_mass_gate = bool(disable_added_mass_inertia_gates or disable_added_mass_gates)
    disable_added_inertia_gate = bool(disable_added_mass_inertia_gates or disable_added_inertia_gates)
    if _is_bernstein_stage(stage_e):
        hydro_gate_upper = _bernstein_gate_upper(stage_e, "hydro_gate_upper")
        added_gate_upper = _bernstein_gate_upper(stage_e, "added_gate_upper")
        model = ForceTorqueBernsteinAblationModel(
            params=copy.deepcopy(params),
            device=device,
            train_wing_added_mass_inertia_gates=False,
            train_wing_hydro_gates=False,
            enable_sweep_gates=True,
            enable_pressure_center_migration=True,
            enable_added_mass_update=True,
            freeze_sweep_geometry=False,
            freeze_added_mass_scaling=disable_added_mass_inertia_gates,
            decouple_gate_eta_from_geometry_freeze=False,
            disable_force_gate=disable_force_gate,
            disable_torque_gate=disable_torque_gate,
            disable_added_mass_gate=disable_added_mass_gate,
            disable_added_inertia_gate=disable_added_inertia_gate,
            linear_added_mass_gate=linear_added_mass_gates,
            linear_added_inertia_gate=linear_added_inertia_gates,
            hydro_gate_upper=hydro_gate_upper,
            added_gate_upper=added_gate_upper,
        )
        _load_bernstein_state(model, stage_e)
        return model

    model_params = copy.deepcopy(params)
    apply_gate_state(model_params, stage_e)
    return ForceTorqueCategoricalAblationModel(
        params=model_params,
        device=device,
        enable_sweep_gates=True,
        enable_pressure_center_migration=True,
        enable_added_mass_update=True,
        freeze_sweep_geometry=False,
        freeze_added_mass_scaling=disable_added_mass_inertia_gates,
        decouple_gate_eta_from_geometry_freeze=False,
        disable_force_gate=disable_force_gate,
        disable_torque_gate=disable_torque_gate,
        disable_added_mass_gate=disable_added_mass_gate,
        disable_added_inertia_gate=disable_added_inertia_gate,
        linear_added_mass_gate=linear_added_mass_gates,
        linear_added_inertia_gate=linear_added_inertia_gates,
    )


def _window_sweep_deg(batch: torch.Tensor) -> float:
    theta_l = batch[:, 19]
    theta_r = batch[:, 20]
    sweep = torch.maximum(torch.abs(theta_l), torch.abs(theta_r))
    return float(torch.max(sweep).detach().cpu().item() * 180.0 / math.pi)


def _group_rmse(err: torch.Tensor) -> Dict[str, float]:
    if err.numel() == 0:
        return {key: float("nan") for key in GROUP_SLICES}
    return {
        key: float(torch.sqrt(torch.mean(err[:, slc] ** 2) + 1e-12).item())
        for key, slc in GROUP_SLICES.items()
    }


def _scalar_metrics(
    pred: torch.Tensor,
    truth: torch.Tensor,
    err_raw: torch.Tensor,
    err_norm: torch.Tensor,
) -> Dict[str, float]:
    if pred.numel() == 0:
        return {
            "num_rollout_samples": 0,
            "mse_raw": float("nan"),
            "rmse_raw": float("nan"),
            "mse_norm": float("nan"),
            "rmse_norm": float("nan"),
            "r2_raw": float("nan"),
        }

    mse_raw = float(torch.mean(err_raw**2).item())
    rmse_raw = float(torch.sqrt(torch.mean(err_raw**2) + 1e-12).item())
    mse_norm = float(torch.mean(err_norm**2).item())
    rmse_norm = float(torch.sqrt(torch.mean(err_norm**2) + 1e-12).item())
    truth_mean = torch.mean(truth, dim=0, keepdim=True)
    sse = torch.sum((pred - truth) ** 2)
    sst = torch.sum((truth - truth_mean) ** 2)
    r2_raw = float((1.0 - sse / (sst + 1e-12)).item())

    return {
        "num_rollout_samples": int(pred.shape[0]),
        "mse_raw": mse_raw,
        "rmse_raw": rmse_raw,
        "mse_norm": mse_norm,
        "rmse_norm": rmse_norm,
        "r2_raw": r2_raw,
    }


@torch.no_grad()
@torch.no_grad()
def _evaluate_variant(
    model,
    traj_list: List[torch.Tensor],
    file_names: List[str],
    *,
    dt: float,
    horizon: int,
    batch_skip: int,
    norm_scale_1_13: Optional[torch.Tensor],
    integrator: str,
    min_sweep_deg: float = 0.0,
    max_windows_per_file: int = 0,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    model.eval()
    step_fn = get_integrator_step(integrator)
    min_sweep_deg = max(0.0, float(min_sweep_deg))
    max_windows_per_file = max(0, int(max_windows_per_file))

    total_pred: List[torch.Tensor] = []
    total_truth: List[torch.Tensor] = []
    total_err_raw: List[torch.Tensor] = []
    total_err_norm: List[torch.Tensor] = []
    window_mse_norm: List[float] = []
    window_rmse_raw: List[float] = []
    window_sweep_deg: List[float] = []
    file_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    total_windows = 0

    for file_idx, (traj, file_name) in enumerate(zip(traj_list, file_names)):
        file_pred: List[torch.Tensor] = []
        file_truth: List[torch.Tensor] = []
        file_err_raw: List[torch.Tensor] = []
        file_err_norm: List[torch.Tensor] = []
        file_window_sweep_deg: List[float] = []
        file_windows = 0

        n = int(traj.shape[0])
        for begin in range(0, max(0, n - horizon + 1), max(1, int(batch_skip))):
            batch = traj[begin : begin + horizon]
            if batch.shape[0] < horizon:
                continue
            sweep_deg = _window_sweep_deg(batch)
            if min_sweep_deg > 0.0 and sweep_deg < min_sweep_deg:
                continue

            state = batch[0].clone()
            preds: List[torch.Tensor] = []
            for k in range(horizon - 1):
                state[13:] = batch[k, 13:]
                state = step_fn(model, state, dt)
                preds.append(state.clone())
            if not preds:
                continue

            pred = torch.stack(preds, dim=0)[:, 1:13]
            truth = batch[1:, 1:13]
            err_raw = pred - truth
            err_norm = err_raw / (norm_scale_1_13 + 1e-12) if norm_scale_1_13 is not None else err_raw

            total_pred.append(pred)
            total_truth.append(truth)
            total_err_raw.append(err_raw)
            total_err_norm.append(err_norm)
            file_pred.append(pred)
            file_truth.append(truth)
            file_err_raw.append(err_raw)
            file_err_norm.append(err_norm)

            mse_raw_window = float(torch.mean(err_raw**2).item())
            mse_norm_window = float(torch.mean(err_norm**2).item())
            rmse_raw_window = float(torch.sqrt(torch.mean(err_raw**2) + 1e-12).item())
            rmse_norm_window = float(torch.sqrt(torch.mean(err_norm**2) + 1e-12).item())
            window_row: Dict[str, object] = {
                "file_index": file_idx,
                "file_name": file_name,
                "begin_index": int(begin),
                "window_index_file": int(file_windows),
                "window_index_global": int(total_windows),
                "horizon": int(horizon),
                "batch_skip": int(batch_skip),
                "min_sweep_deg_filter": min_sweep_deg,
                "sweep_deg_max": sweep_deg,
                "mse_raw": mse_raw_window,
                "mse_norm": mse_norm_window,
                "rmse_raw": rmse_raw_window,
                "rmse_norm": rmse_norm_window,
            }
            for key, slc in GROUP_SLICES.items():
                window_row[f"mse_raw_{key}"] = float(torch.mean(err_raw[:, slc] ** 2).item())
                window_row[f"mse_norm_{key}"] = float(torch.mean(err_norm[:, slc] ** 2).item())
                window_row[f"rmse_raw_{key}"] = float(torch.sqrt(torch.mean(err_raw[:, slc] ** 2) + 1e-12).item())
                window_row[f"rmse_norm_{key}"] = float(torch.sqrt(torch.mean(err_norm[:, slc] ** 2) + 1e-12).item())
            window_rows.append(window_row)

            file_windows += 1
            total_windows += 1
            window_mse_norm.append(mse_norm_window)
            window_rmse_raw.append(rmse_raw_window)
            window_sweep_deg.append(sweep_deg)
            file_window_sweep_deg.append(sweep_deg)

            if max_windows_per_file and file_windows >= max_windows_per_file:
                break

        empty = torch.empty((0, 12), dtype=traj.dtype, device=traj.device)
        if file_pred:
            pred_all = torch.cat(file_pred, dim=0)
            truth_all = torch.cat(file_truth, dim=0)
            err_raw_all = torch.cat(file_err_raw, dim=0)
            err_norm_all = torch.cat(file_err_norm, dim=0)
        else:
            pred_all = truth_all = err_raw_all = err_norm_all = empty

        metrics = _scalar_metrics(pred_all, truth_all, err_raw_all, err_norm_all)
        group_raw = _group_rmse(err_raw_all)
        group_norm = _group_rmse(err_norm_all)
        row: Dict[str, object] = {
            "file_index": file_idx,
            "file_name": file_name,
            "num_windows": file_windows,
            "min_sweep_deg_filter": min_sweep_deg,
            "sweep_deg_window_mean": (
                float(sum(file_window_sweep_deg) / len(file_window_sweep_deg))
                if file_window_sweep_deg
                else float("nan")
            ),
            "sweep_deg_window_max": float(max(file_window_sweep_deg)) if file_window_sweep_deg else float("nan"),
            **metrics,
        }
        for key in GROUP_SLICES:
            row[f"rmse_raw_{key}"] = group_raw[key]
            row[f"rmse_norm_{key}"] = group_norm[key]
        file_rows.append(row)

    if not total_pred:
        raise ValueError(f"No valid rollout windows were generated for horizon={horizon}.")

    pred_all = torch.cat(total_pred, dim=0)
    truth_all = torch.cat(total_truth, dim=0)
    err_raw_all = torch.cat(total_err_raw, dim=0)
    err_norm_all = torch.cat(total_err_norm, dim=0)
    metrics = _scalar_metrics(pred_all, truth_all, err_raw_all, err_norm_all)
    group_raw = _group_rmse(err_raw_all)
    group_norm = _group_rmse(err_norm_all)

    summary: Dict[str, object] = {
        "num_windows": total_windows,
        "min_sweep_deg_filter": min_sweep_deg,
        "sweep_deg_window_mean": float(sum(window_sweep_deg) / len(window_sweep_deg)) if window_sweep_deg else float("nan"),
        "sweep_deg_window_max": float(max(window_sweep_deg)) if window_sweep_deg else float("nan"),
        "traj_mse_windowed": float(sum(window_mse_norm) / len(window_mse_norm)),
        "window_rmse_raw_mean": float(sum(window_rmse_raw) / len(window_rmse_raw)),
        "window_rmse_raw_std": float(torch.tensor(window_rmse_raw, dtype=torch.double).std(unbiased=False).item()),
        **metrics,
    }
    for key in GROUP_SLICES:
        summary[f"rmse_raw_{key}"] = group_raw[key]
        summary[f"rmse_norm_{key}"] = group_norm[key]
    return summary, file_rows, window_rows


def _save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_value_keys() -> List[str]:
    keys = [
        "traj_mse_windowed",
        "mse_raw",
        "rmse_raw",
        "mse_norm",
        "rmse_norm",
        "r2_raw",
        "window_rmse_raw_mean",
        "window_rmse_raw_std",
    ]
    for group in GROUP_SLICES:
        keys.append(f"rmse_raw_{group}")
        keys.append(f"rmse_norm_{group}")
    return keys


def _build_delta_rows(
    rows: List[Dict[str, object]],
    key_fields: Sequence[str],
    *,
    reference_variant: str = "full_model",
) -> List[Dict[str, object]]:
    metric_keys = _metric_value_keys()
    by_key: Dict[Tuple[object, ...], Dict[str, Dict[str, object]]] = {}

    for row in rows:
        key = tuple(row.get(k) for k in key_fields)
        by_key.setdefault(key, {})[str(row["variant"])] = row

    out: List[Dict[str, object]] = []
    for _, variant_map in by_key.items():
        ref = variant_map.get(reference_variant)
        if ref is None:
            continue
        for variant, row in variant_map.items():
            merged = {field: row.get(field) for field in key_fields}
            merged["variant"] = variant
            merged["reference_variant"] = reference_variant
            for metric in metric_keys:
                value = float(row.get(metric, float("nan")))
                ref_value = float(ref.get(metric, float("nan")))
                if math.isfinite(value) and math.isfinite(ref_value):
                    delta = value - ref_value
                    ratio = value / ref_value if abs(ref_value) > 1e-18 else float("nan")
                else:
                    delta = float("nan")
                    ratio = float("nan")
                merged[metric] = value
                merged[f"{metric}_{reference_variant}"] = ref_value
                merged[f"{metric}_delta_vs_{reference_variant}"] = delta
                merged[f"{metric}_ratio_vs_{reference_variant}"] = ratio
            out.append(merged)
    return out


def _best_variant(rows: List[Dict[str, object]], metric_key: str) -> Tuple[Optional[str], float]:
    best_name: Optional[str] = None
    best_value = float("inf")
    for row in rows:
        value = float(row.get(metric_key, float("nan")))
        if math.isfinite(value) and value < best_value:
            best_name = str(row["variant"])
            best_value = value
    return best_name, best_value



SWEEP_COLUMNS = (19, 20, 21, 22, 23, 24)
PROJECT_ROOT = Path(THIS_DIR).parents[1]
DEFAULT_INDEX_CSV = PROJECT_ROOT / "data" / "index" / "index.csv"
DEFAULT_NORM_STATS_JSON = PROJECT_ROOT / "config" / "norm_stats_minmax_local.json"
HORIZON = 360
DT = 1.0 / 90.0
INTEGRATOR = "rk2"

LABEL_TO_FT_VARIANT = {
    "full_model": "full_model",
    "no_force_scale": "no_force_gate",
    "no_torque_scale": "no_torque_gate",
    "no_add_mass_scale": "no_added_mass_gate",
    "no_add_inertia_scale": "no_added_inertia_gate",
}

SCALE_VARIANTS = [
    "full_model",
    "no_force_scale",
    "no_torque_scale",
    "no_add_mass_scale",
    "no_add_inertia_scale",
]
MULTI_VS_SINGLE_VARIANTS = ["full_model", "zero_sweep"]
_G = {}


def _required_path(value: Optional[str], env_name: str) -> str:
    if value:
        return value
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    raise ValueError(f"Pass --{env_name.lower().replace('_', '-')} or set {env_name}.")


def _parse_scheme_name(data_file_name: str):
    import re

    match = re.match(r"^(\d{4})_(\d+)_(\d+)-", Path(data_file_name).name)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def load_scheme41_window_index(index_csv: Path, total_windows: int) -> pd.DataFrame:
    index_df = pd.read_csv(index_csv)
    rows = []
    for _, source in index_df.iterrows():
        data_file_name = str(source["data_file_name"]).replace("\\", "/")
        parsed = _parse_scheme_name(data_file_name)
        if parsed is None:
            continue
        _, condition, trial = parsed
        if not (data_file_name.startswith("processed_mocap/stage_a_to_e/") and 44 <= condition <= 61 and trial == 4):
            continue
        row_begin = int(source["row_begin"])
        row_end = int(source["row_end"])
        length = row_end - row_begin + 1
        if length <= 0 or length % HORIZON != 0:
            raise ValueError(f"{data_file_name}@{row_begin}:{row_end} is not a positive multiple of {HORIZON}")
        for local_idx, begin in enumerate(range(row_begin, row_end + 1, HORIZON)):
            rows.append(
                {
                    "window_id": f"{Path(data_file_name).stem}__{begin:05d}",
                    "data_file_name": data_file_name,
                    "file_name": Path(data_file_name).name,
                    "trajectory_index": len({row["data_file_name"] for row in rows}),
                    "window_index_file": local_idx,
                    "window_index_global": len(rows),
                    "row_begin": begin,
                    "row_end": begin + HORIZON - 1,
                }
            )
    if len(rows) != int(total_windows):
        raise ValueError(f"Expected {total_windows} windows, found {len(rows)} in {index_csv}")
    return pd.DataFrame(rows)


def _zero_sweep_trajectories(traj_list: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for traj in traj_list:
        copied = traj.clone()
        copied[:, list(SWEEP_COLUMNS)] = 0.0
        out.append(copied)
    return out


def _build_stage_bd_model(*, params: Dict, device: str) -> AWUGModelFromTex:
    return AWUGModelFromTex(
        params=copy.deepcopy(params),
        device=device,
        enable_sweep_gates=False,
        enable_pressure_center_migration=True,
        enable_added_mass_update=True,
        freeze_sweep_geometry=False,
        freeze_added_mass_scaling=False,
        decouple_gate_eta_from_geometry_freeze=False,
    )


def worker_init(
    data_file_names: list[str],
    labels: list[str],
    input_b_json: str,
    input_d_json: str,
    input_e_json: str,
    norm_stats_json: str,
) -> None:
    torch.set_num_threads(1)
    stage_b = load_stage_json(input_b_json)
    stage_d = load_stage_json(input_d_json)
    stage_e = load_stage_json(input_e_json)
    params = _build_model_params(stage_b, stage_d)
    norm_scale = load_norm_scale(norm_stats_json, mode="minmax", device="cpu")
    available = {name: spec for name, spec in VARIANT_SPECS}

    models = {}
    for label in labels:
        if label == "zero_sweep":
            models[label] = _build_stage_bd_model(params=params, device="cpu")
        else:
            models[label] = _build_variant_model(
                device="cpu",
                params=params,
                stage_e=stage_e,
                **copy.deepcopy(available[LABEL_TO_FT_VARIANT[label]]),
            )

    frames = {name: pd.read_excel(PROJECT_ROOT / "data" / name) for name in sorted(set(data_file_names))}
    _G.clear()
    _G.update(models=models, norm_scale=norm_scale, frames=frames, labels=labels)


def eval_one_window(task: dict) -> dict:
    data_file_name = str(task["data_file_name"])
    row_begin = int(task["row_begin"])
    df = _G["frames"][data_file_name].iloc[row_begin : row_begin + HORIZON].copy().reset_index(drop=True)
    if len(df) != HORIZON:
        raise ValueError(f"{data_file_name} row_begin={row_begin} produced {len(df)} rows, expected {HORIZON}")
    df["t"] = [i * DT for i in range(len(df))]
    from SI_awug_crba_utils import build_state_tensor

    traj = build_state_tensor(df, device="cpu", sample_step=1)

    out = dict(task)
    for label in _G["labels"]:
        traj_list = _zero_sweep_trajectories([traj]) if label == "zero_sweep" else [traj]
        summary, _, _ = _evaluate_variant(
            _G["models"][label],
            traj_list,
            [str(task["window_id"])],
            dt=DT,
            horizon=HORIZON,
            batch_skip=HORIZON,
            norm_scale_1_13=_G["norm_scale"],
            integrator=INTEGRATOR,
            min_sweep_deg=0.0,
            max_windows_per_file=0,
        )
        out[f"{label}_mse_norm"] = float(summary["mse_norm"])
    return out


def summarize(wide: pd.DataFrame, variants: Sequence[str]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        metric_col = f"{variant}_mse_norm"
        trajectory_means = wide.groupby("data_file_name", sort=True)[metric_col].mean()
        rows.append(
            {
                "variant": variant,
                "window_mean_mse_norm": float(wide[metric_col].mean()),
                "window_median_mse_norm": float(wide[metric_col].median()),
                "trajectory_mean_mse_norm": float(trajectory_means.mean()),
                "trajectory_median_mse_norm": float(trajectory_means.median()),
            }
        )
    return pd.DataFrame(rows)


def parse_scheme_args(description: str, default_out_dir: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-b-json", default=None)
    parser.add_argument("--input-d-json", default=None)
    parser.add_argument("--input-e-json", default=None)
    parser.add_argument("--total-windows", type=int, default=41)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--index-csv", default=str(DEFAULT_INDEX_CSV))
    parser.add_argument("--norm-stats-json", default=str(DEFAULT_NORM_STATS_JSON))
    parser.add_argument("--out-dir", default=str(THIS_DIR and (Path(THIS_DIR) / default_out_dir)))
    return parser.parse_args()


def run_scheme41_summary(*, variants: Sequence[str], output_csv: str, description: str, default_out_dir: str) -> Path:
    from concurrent.futures import ProcessPoolExecutor

    args = parse_scheme_args(description, default_out_dir)
    input_b_json = _required_path(args.input_b_json, "INPUT_B_JSON")
    input_d_json = _required_path(args.input_d_json, "INPUT_D_JSON")
    input_e_json = _required_path(args.input_e_json, "INPUT_E_JSON")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_index = load_scheme41_window_index(Path(args.index_csv), int(args.total_windows))
    tasks = window_index.to_dict("records")
    data_file_names = sorted(window_index["data_file_name"].unique().tolist())

    print(
        f"[setup] windows={len(window_index)} trajectories={len(data_file_names)} "
        f"variants={list(variants)} workers={args.workers}",
        flush=True,
    )

    results = []
    init_args = (
        data_file_names,
        list(variants),
        input_b_json,
        input_d_json,
        input_e_json,
        str(args.norm_stats_json),
    )
    if int(args.workers) <= 1:
        worker_init(*init_args)
        for idx, task in enumerate(tasks, 1):
            results.append(eval_one_window(task))
            if idx % 5 == 0:
                print(f"[eval] {idx}/{len(tasks)} windows", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers), initializer=worker_init, initargs=init_args) as ex:
            for idx, result in enumerate(ex.map(eval_one_window, tasks, chunksize=1), 1):
                results.append(result)
                if idx % 5 == 0:
                    print(f"[eval] {idx}/{len(tasks)} windows", flush=True)

    wide = pd.DataFrame(results).sort_values("window_index_global").reset_index(drop=True)
    output_path = out_dir / output_csv
    summarize(wide, variants).to_csv(output_path, index=False)
    print(f"[done] wrote {output_path}", flush=True)
    return output_path


def main() -> None:
    run_scheme41_summary(
        variants=SCALE_VARIANTS,
        output_csv="scale_ablation_summary.csv",
        description="Evaluate Scheme41 scale ablations.",
        default_out_dir="scale_ablation_results",
    )


if __name__ == "__main__":
    main()
