from __future__ import annotations


import argparse
import os
import sys
from typing import Dict, Sequence

import numpy as np
import torch

from SI_awug_crba_utils import (
    OLD_STAGE_B_JSON,
    apply_body_state,
    body_symmetry_matrix,
    bounded_lsq,
    build_body_model,
    build_model,
    build_params_with_body,
    clone_base_params,
    collect_body_residual_system,
    load_stage_data,
    load_json,
    loss_delta,
    nan_loss_dict,
    rollout_loss_eval,
    save_json,
    signed_summary,
    start_stage,
)


# Original y-source independent names:
# THETA3_NAMES = ("X_u", "X_uu", "Y_v", "Y_vv", "Y_r", "Y_rr")
# THETA4_NAMES = ("K_p", "K_pp", "M_q", "M_qq", "M_w", "M_ww")
THETA3_NAMES = ("X_u", "X_uu", "Z_w", "Z_ww", "Z_q", "Z_qq")
THETA4_NAMES = ("K_p", "K_pp", "N_r", "N_rr", "N_v", "N_vv")


def parse_vector(text: str, *, name: str, size: int = 6) -> np.ndarray:
    """Parse a comma-separated numeric vector and enforce its expected length."""
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != size:
        raise ValueError(f"{name} expects {size} comma-separated values, got {len(vals)}: {text}")
    return np.asarray(vals, dtype=np.float64)


def vector_dict(names: Sequence[str], values: np.ndarray) -> Dict[str, float]:
    """Attach readable coefficient names to a vector for JSON diagnostics."""
    return {name: float(value) for name, value in zip(names, values)}


def solve_selective_stage_a(model, traj_list, args):
    """Build body residual equations and solve bounded LS for selective Stage A."""
    x3, x4, y3, y4, row_stats = collect_body_residual_system(
        model,
        traj_list,
        min_linear_speed=args.min_linear_speed,
        min_angular_speed=args.min_angular_speed,
        verbose=args.verbose,
    )
    s = body_symmetry_matrix(device=model.device, dtype=model.dtype)
    a3 = x3 @ s
    a4 = x4 @ s

    theta3_lower = parse_vector(args.theta3_lower, name="theta3_lower")
    theta3_upper = parse_vector(args.theta3_upper, name="theta3_upper")
    theta4_lower = parse_vector(args.theta4_lower, name="theta4_lower")
    theta4_upper = parse_vector(args.theta4_upper, name="theta4_upper")
    if np.any(theta3_lower > theta3_upper):
        raise ValueError("theta3_lower must be <= theta3_upper elementwise")
    if np.any(theta4_lower > theta4_upper):
        raise ValueError("theta4_lower must be <= theta4_upper elementwise")

    theta3_ind, meta3 = bounded_lsq(
        a3,
        y3,
        lower=theta3_lower,
        upper=theta3_upper,
        block_dim=3,
        weighted=(not args.unweighted),
        col_normalize=args.ls_col_normalize,
        ridge_lambda=args.ridge_lambda,
    )
    theta4_ind, meta4 = bounded_lsq(
        a4,
        y4,
        lower=theta4_lower,
        upper=theta4_upper,
        block_dim=3,
        weighted=(not args.unweighted),
        col_normalize=args.ls_col_normalize,
        ridge_lambda=args.ridge_lambda,
    )
    theta3 = s @ theta3_ind
    theta4 = s @ theta4_ind
    meta = {
        "row_stats": row_stats,
        "theta_3_solver": meta3,
        "theta_4_solver": meta4,
        "theta3_names": THETA3_NAMES,
        "theta4_names": THETA4_NAMES,
        "theta3_lower": vector_dict(THETA3_NAMES, theta3_lower),
        "theta3_upper": vector_dict(THETA3_NAMES, theta3_upper),
        "theta4_lower": vector_dict(THETA4_NAMES, theta4_lower),
        "theta4_upper": vector_dict(THETA4_NAMES, theta4_upper),
        "constraint_interpretation": (
            "Selective independent-body bounds: main damping terms are constrained negative; "
            "cross-coupling terms are allowed wider signed ranges; effective 10-vectors preserve "
            "paired opposite-sign body symmetry with z-direction coefficients as the source."
        ),
    }
    return theta3, theta4, theta3_ind, theta4_ind, meta


def build_parser() -> argparse.ArgumentParser:
    """Define all knobs used by selective Stage A."""
    base_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
        description="Stage A body CLS with selective per-parameter bounds on new ProcessedData workbooks."
    )
    parser.add_argument("--train-files", default="")
    parser.add_argument("--test-files", default="")
    parser.add_argument("--old-b-json", default=OLD_STAGE_B_JSON)
    parser.add_argument("--out-dir", default="log/si_awug_stage_a")
    parser.add_argument("--body-symmetry-mode", default="independent_negative")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--dt-base", type=float, default=1.0 / 90.0)
    parser.add_argument("--norm-mode", choices=["none", "minmax", "zscore"], default="minmax")
    parser.add_argument("--norm-stats-json", default=os.path.join(base_dir, "norm_stats_minmax.json"))
    parser.add_argument("--loss-weight-mode", choices=["none", "variance", "task", "variance_task"], default="none")
    parser.add_argument("--variance-weights-json", default="")
    parser.add_argument("--task-weights-spec", default="")
    parser.add_argument("--theta3-lower", default="-200,-500,-200,-500,-100,-200")
    parser.add_argument("--theta3-upper", default="50,200,50,200,100,200")
    parser.add_argument("--theta4-lower", default="-50,-100,-50,-200,-100,-300")
    parser.add_argument("--theta4-upper", default="20,50,20,100,100,300")
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--min-linear-speed", type=float, default=0.01)
    parser.add_argument("--min-angular-speed", type=float, default=0.02)
    parser.add_argument("--batch-size-max", type=int, default=360)
    parser.add_argument("--batch-skip", type=int, default=360)
    parser.add_argument("--unweighted", action="store_true")
    parser.add_argument("--ls-col-normalize", action="store_true")
    parser.add_argument("--skip-old-eval", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_dir = os.path.dirname(__file__)
    run_dir, log_path = start_stage(
        "stage_a",
        args.out_dir,
        sys.argv,
        base_dir=base_dir,
    )

    data = load_stage_data(
        train_files_arg=args.train_files,
        test_files_arg=args.test_files,
        device=args.device,
        sample_step=args.sample_step,
        dt_base=args.dt_base,
        norm_mode=args.norm_mode,
        norm_stats_json=args.norm_stats_json,
        loss_weight_mode=args.loss_weight_mode,
        variance_weights_json=args.variance_weights_json,
        task_weights_spec=args.task_weights_spec,
        base_dir=base_dir,
        include_test=True,
    )
    print(f"[data] train_files={len(data.train_files)} test_files={len(data.test_files)} dt={data.dt:.12g}")
    print(
        f"[data] train_rows={sum(int(t.shape[0]) for t in data.train_traj)} "
        f"test_rows={sum(int(t.shape[0]) for t in data.test_traj)}"
    )

    if args.skip_old_eval:
        print("[old] skipping old Stage-B loss eval (--skip-old-eval).", flush=True)
        old_stage_b = None
        old_loss = nan_loss_dict()
    else:
        old_stage_b = load_json(args.old_b_json)
        old_params = apply_body_state(clone_base_params(), old_stage_b)
        old_model = build_model(device=args.device, params=old_params)
        print("[old] evaluating old Stage-B loss on new data...", flush=True)
        old_loss = rollout_loss_eval(old_model, data, args)
        print(
            f"[old] test_traj_loss={old_loss['test_traj_loss']:.6e}"
        )

    model = build_body_model(args)
    print("[stage_a_selective] collecting residual rows and solving selective-bound CLS...", flush=True)
    theta3, theta4, theta3_ind, theta4_ind, solver_meta = solve_selective_stage_a(model, data.train_traj, args)
    stage_a_params = build_params_with_body(theta3, theta4)
    stage_a_model = build_body_model(args, params=stage_a_params)
    print("[stage_a_selective] evaluating selective Stage-A loss...", flush=True)
    stage_a_loss = rollout_loss_eval(stage_a_model, data, args)

    print(f"[stage_a_selective] Theta_3={theta3.detach().cpu().reshape(-1).tolist()}")
    print(f"[stage_a_selective] Theta_4={theta4.detach().cpu().reshape(-1).tolist()}")
    print(
        f"[stage_a_selective] test_traj_loss={stage_a_loss['test_traj_loss']:.6e}"
    )

    result = {
        "meta": {
            "stage": "stage_a",
            "log_path": log_path,
            "old_b_json": os.path.abspath(args.old_b_json),
            "train_files": data.train_files,
            "test_files": data.test_files,
            "dt": data.dt,
            "dt_base": args.dt_base,
            "sample_step": args.sample_step,
            "device": args.device,
            "ridge_lambda": args.ridge_lambda,
            "weighted": not args.unweighted,
            "ls_col_normalize": args.ls_col_normalize,
            "min_linear_speed": args.min_linear_speed,
            "min_angular_speed": args.min_angular_speed,
            "batch_size_max": args.batch_size_max,
            "batch_skip": args.batch_skip,
            "norm_mode": args.norm_mode,
            "norm_stats_json": args.norm_stats_json,
            "loss_weight_mode": args.loss_weight_mode,
            "skip_old_eval": bool(args.skip_old_eval),
        },
        "scheme": solver_meta["constraint_interpretation"],
        # Compatibility with the original Stage-B entrypoint, which reads
        # top-level body_hydro_params from a Stage-A JSON.
        "body_hydro_params": {
            "Theta_3": theta3,
            "Theta_4": theta4,
        },
        "stage_a_selective": {
            "body_hydro_params": {
                "Theta_3": theta3,
                "Theta_4": theta4,
                "Theta_3_independent": theta3_ind,
                "Theta_4_independent": theta4_ind,
            },
            "sign_summary": {
                "Theta_3": signed_summary(theta3, independent_len=6),
                "Theta_4": signed_summary(theta4, independent_len=6),
            },
            "solver": solver_meta,
            "loss": stage_a_loss,
            "loss_delta_vs_old_b": loss_delta(stage_a_loss, old_loss),
        },
        "old_stage_b_on_newdata": {
            "skipped": bool(args.skip_old_eval),
            "body_hydro_params": old_stage_b.get("body_hydro_params", {}) if old_stage_b else {},
            "body_added_mass_inertia": old_stage_b.get("body_added_mass_inertia", {}) if old_stage_b else {},
            "loss": old_loss,
        },
    }

    out_path = os.path.join(run_dir, "stage_a_result.json")
    save_json(out_path, result)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
