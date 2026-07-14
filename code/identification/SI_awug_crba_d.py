from __future__ import annotations


import argparse
import os
import sys

import torch

from SI_awug_crba_utils import (
    GdDefaults,
    apply_body_state,
    apply_wing_ls_state,
    add_gd_arguments,
    build_constrained_forward,
    build_model,
    clone_base_params,
    build_gd_meta,
    load_stage_data,
    load_stage_json,
    make_constraint,
    make_positive_diagonal_matrix_constraint,
    plot_test_rollout_windows,
    pretty_vector,
    rollout_loss_eval,
    run_gd_training,
    save_json,
    start_stage,
)


THETA_5_NAMES = ("C_D0", "Delta_C_Dalpha", "C_Sbeta", "C_Lalpha")
THETA_6_NAMES = ("C_xbeta", "C_xp_d", "C_m0", "C_malpha", "C_mq_d", "C_zbeta", "C_zr_d")
M_AL_DIAG_NAMES = ("Mxx", "Myy", "Mzz")
I_AL_DIAG_NAMES = ("Ixx", "Iyy", "Izz")


def _resolve_bound_index(key: str, names: tuple[str, ...], *, label: str) -> int:
    """Resolve a bound-spec key by coefficient name or zero-based index."""
    key = key.strip()
    if key in names:
        return names.index(key)
    try:
        idx = int(key)
    except ValueError as exc:
        raise ValueError(f"{label}: unknown coefficient {key!r}; expected one of {names}") from exc
    if idx < 0 or idx >= len(names):
        raise ValueError(f"{label}: index {idx} out of range 0..{len(names)-1}")
    return idx


def _apply_elementwise_bound_spec(
    constraint,
    spec: str,
    names: tuple[str, ...],
    *,
    eps: float,
    label: str,
) -> list[dict]:
    """Override selected tensor bounds with per-coefficient relative or absolute specs.

    Spec examples:
        C_Lalpha:abs:0:0.2
        C_mq_d:relative:0.02
        4:fixed:0

    Multiple entries may be separated by commas or semicolons.
    """
    text = (spec or "").strip()
    if not text:
        return []
    entries = [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]
    init = constraint.init.reshape(-1)
    lo_flat = constraint._lo.reshape(-1)
    hi_flat = constraint._hi.reshape(-1)
    raw_flat = constraint.raw.reshape(-1)
    updates: list[dict] = []
    with torch.no_grad():
        for entry in entries:
            parts = [p.strip() for p in entry.split(":")]
            if len(parts) < 2:
                raise ValueError(f"{label}: bad bound entry {entry!r}")
            idx = _resolve_bound_index(parts[0], names, label=label)
            mode = parts[1].lower()
            center = init[idx]

            if mode in {"relative", "rel"}:
                if len(parts) != 3:
                    raise ValueError(f"{label}: relative entry must be name:relative:ratio, got {entry!r}")
                ratio = float(parts[2])
                lo = center * (1.0 - ratio)
                hi = center * (1.0 + ratio)
            elif mode in {"abs", "absolute"}:
                if len(parts) != 4:
                    raise ValueError(f"{label}: abs entry must be name:abs:lower:upper, got {entry!r}")
                lo = torch.as_tensor(float(parts[2]), dtype=center.dtype, device=center.device)
                hi = torch.as_tensor(float(parts[3]), dtype=center.dtype, device=center.device)
            elif mode in {"abs_scaled", "scaled"}:
                if len(parts) != 3:
                    raise ValueError(f"{label}: abs_scaled entry must be name:abs_scaled:ratio, got {entry!r}")
                ratio = float(parts[2])
                mag = torch.maximum(torch.abs(center), torch.as_tensor(eps, dtype=center.dtype, device=center.device))
                lo = center - ratio * mag
                hi = center + ratio * mag
            elif mode == "fixed":
                if len(parts) == 2:
                    lo = center
                    hi = center
                elif len(parts) == 3:
                    lo = torch.as_tensor(float(parts[2]), dtype=center.dtype, device=center.device)
                    hi = lo
                else:
                    raise ValueError(f"{label}: fixed entry must be name:fixed or name:fixed:value, got {entry!r}")
            else:
                raise ValueError(f"{label}: unsupported bound mode {mode!r} in {entry!r}")

            lo_sorted = torch.minimum(lo, hi)
            hi_sorted = torch.maximum(lo, hi)
            lo_flat[idx].copy_(lo_sorted)
            hi_flat[idx].copy_(hi_sorted)
            raw_flat[idx].copy_(torch.maximum(torch.minimum(raw_flat[idx], hi_sorted), lo_sorted))
            updates.append(
                {
                    "name": names[idx],
                    "index": idx,
                    "mode": mode,
                    "init": float(center.detach().cpu()),
                    "lower": float(lo_sorted.detach().cpu()),
                    "upper": float(hi_sorted.detach().cpu()),
                }
            )
    return updates


def _apply_positive_diagonal_bound_spec(
    constraint,
    spec: str,
    names: tuple[str, ...],
    *,
    eps: float,
    label: str,
) -> list[dict]:
    """Override selected positive diagonal-matrix bounds.

    Only diagonal entries are trainable in Stage D; off-diagonal entries remain
    zero. Spec syntax matches `_apply_elementwise_bound_spec`.
    """
    text = (spec or "").strip()
    if not text:
        return []
    entries = [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]
    init_diag = torch.diagonal(constraint.init).reshape(-1)
    lo_diag = constraint._lo_diag.reshape(-1)
    hi_diag = constraint._hi_diag.reshape(-1)
    raw_diag = constraint.raw.reshape(-1)
    floor = torch.as_tensor(
        constraint.min_positive,
        dtype=init_diag.dtype,
        device=init_diag.device,
    )
    updates: list[dict] = []
    with torch.no_grad():
        for entry in entries:
            parts = [p.strip() for p in entry.split(":")]
            if len(parts) < 2:
                raise ValueError(f"{label}: bad bound entry {entry!r}")
            idx = _resolve_bound_index(parts[0], names, label=label)
            mode = parts[1].lower()
            center = init_diag[idx]

            if mode in {"relative", "rel"}:
                if len(parts) != 3:
                    raise ValueError(f"{label}: relative entry must be name:relative:ratio, got {entry!r}")
                ratio = float(parts[2])
                lo = center * (1.0 - ratio)
                hi = center * (1.0 + ratio)
            elif mode in {"abs", "absolute"}:
                if len(parts) != 4:
                    raise ValueError(f"{label}: abs entry must be name:abs:lower:upper, got {entry!r}")
                lo = torch.as_tensor(float(parts[2]), dtype=center.dtype, device=center.device)
                hi = torch.as_tensor(float(parts[3]), dtype=center.dtype, device=center.device)
            elif mode in {"abs_scaled", "scaled"}:
                if len(parts) != 3:
                    raise ValueError(f"{label}: abs_scaled entry must be name:abs_scaled:ratio, got {entry!r}")
                ratio = float(parts[2])
                mag = torch.maximum(torch.abs(center), torch.as_tensor(eps, dtype=center.dtype, device=center.device))
                lo = center - ratio * mag
                hi = center + ratio * mag
            elif mode == "fixed":
                if len(parts) == 2:
                    lo = center
                    hi = center
                elif len(parts) == 3:
                    lo = torch.as_tensor(float(parts[2]), dtype=center.dtype, device=center.device)
                    hi = lo
                else:
                    raise ValueError(f"{label}: fixed entry must be name:fixed or name:fixed:value, got {entry!r}")
            else:
                raise ValueError(f"{label}: unsupported bound mode {mode!r} in {entry!r}")

            lo_sorted = torch.minimum(lo, hi)
            hi_sorted = torch.maximum(lo, hi)
            lo_pos = torch.maximum(lo_sorted, floor)
            hi_pos = torch.maximum(hi_sorted, lo_pos)
            lo_diag[idx].copy_(lo_pos)
            hi_diag[idx].copy_(hi_pos)
            raw_diag[idx].copy_(torch.maximum(torch.minimum(raw_diag[idx], hi_pos), lo_pos))
            updates.append(
                {
                    "name": names[idx],
                    "index": idx,
                    "mode": mode,
                    "init": float(center.detach().cpu()),
                    "lower": float(lo_pos.detach().cpu()),
                    "upper": float(hi_pos.detach().cpu()),
                }
            )
        constraint._lo = torch.diag(constraint._lo_diag).clone()
        constraint._hi = torch.diag(constraint._hi_diag).clone()
    return updates


def _stage_d_constrained_lrs(args) -> dict[str, float]:
    """Collect optional per-group learning rates for Stage D constrained raws."""
    items = {
        "M_al": args.lr_m_al,
        "I_al": args.lr_i_al,
        "Theta_5": args.lr_theta_5,
        "Theta_6": args.lr_theta_6,
    }
    lrs: dict[str, float] = {}
    for name, value in items.items():
        if value is None:
            continue
        lr_value = float(value)
        if lr_value <= 0:
            raise ValueError(f"{name} learning rate must be positive, got {lr_value}")
        lrs[name] = lr_value
    return lrs


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage D command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Stage D constrained gradient descent for wing inertia and wing hydrodynamic coefficients."
            " / Constrained GD for wing inertia and wing hydrodynamic coefficients."
        )
    )
    parser.add_argument("--input-b-json", required=True, help="Path to stage_b_result.json.")
    parser.add_argument("--input-c-json", required=True, help="Path to stage_c_result.json.")
    parser.add_argument("--train-files", required=True)
    parser.add_argument("--test-files", required=True)
    parser.add_argument("--out-dir", default="log/si_awug_crba")

    parser.add_argument("--ratio-m-al", type=float, default=0.5)
    parser.add_argument("--ratio-i-al", type=float, default=0.5)
    parser.add_argument("--ratio-theta-5", type=float, default=0.5)
    parser.add_argument("--ratio-theta-6", type=float, default=0.5)
    parser.add_argument("--lr-m-al", type=float, default=None, help="Optional Adam lr for M_al raw parameters.")
    parser.add_argument("--lr-i-al", type=float, default=None, help="Optional Adam lr for I_al raw parameters.")
    parser.add_argument("--lr-theta-5", type=float, default=None, help="Optional Adam lr for Theta_5 raw parameters.")
    parser.add_argument("--lr-theta-6", type=float, default=None, help="Optional Adam lr for Theta_6 raw parameters.")
    parser.add_argument(
        "--m-al-bound-spec",
        default="",
        help=(
            "Per-diagonal M_al bound overrides. Entries are separated by commas or semicolons. "
            "Formats: name:relative:ratio, name:abs:lower:upper, name:abs_scaled:ratio, name:fixed[:value]. "
            "Names: Mxx, Myy, Mzz; indices 0,1,2 are also accepted."
        ),
    )
    parser.add_argument(
        "--i-al-bound-spec",
        default="",
        help=(
            "Per-diagonal I_al bound overrides. Entries are separated by commas or semicolons. "
            "Formats: name:relative:ratio, name:abs:lower:upper, name:abs_scaled:ratio, name:fixed[:value]. "
            "Names: Ixx, Iyy, Izz; indices 0,1,2 are also accepted."
        ),
    )
    parser.add_argument(
        "--theta5-bound-spec",
        default="",
        help=(
            "Per-coefficient Theta_5 bound overrides. Entries are separated by commas or semicolons. "
            "Formats: name:relative:ratio, name:abs:lower:upper, name:abs_scaled:ratio, name:fixed[:value]. "
            "Names: C_D0, Delta_C_Dalpha, C_Sbeta, C_Lalpha."
        ),
    )
    parser.add_argument(
        "--theta6-bound-spec",
        default="",
        help=(
            "Per-coefficient Theta_6 bound overrides. Entries are separated by commas or semicolons. "
            "Formats: name:relative:ratio, name:abs:lower:upper, name:abs_scaled:ratio, name:fixed[:value]. "
            "Names: C_xbeta, C_xp_d, C_m0, C_malpha, C_mq_d, C_zbeta, C_zr_d."
        ),
    )
    parser.add_argument("--band", choices=["relative", "abs_scaled"], default="abs_scaled")
    parser.add_argument(
        "--theta-band",
        choices=["relative", "abs_scaled", "abs_symmetric"],
        default=None,
        help=(
            "Optional bound mode for Theta_5/Theta_6 only. "
            "abs_symmetric means [-ratio*abs(init), +ratio*abs(init)]."
        ),
    )
    parser.add_argument("--band-eps", type=float, default=1e-8)
    parser.add_argument(
        "--constraint-mode",
        choices=["project"],
        default="project",
        help="Use projected box constraints for all bounded Stage D parameters.",
    )
    parser.add_argument("--init-check", action="store_true")

    add_gd_arguments(
        parser,
        defaults=GdDefaults(
            epochs=30,
            lr=1e-3,
            grad_clip=1.0,
            dt_base=1.0 / 90.0,
            batch_size_min=8,
            batch_size_max=64,
            batch_size_schedule="linear",
            batch_skip=1,
            norm_mode="none",
            norm_stats_json="norm_stats_minmax.json",
            loss_weight_mode="none",
            variance_weights_json=None,
            task_weights_spec=None,
            train_progress_every=30,
            max_windows_per_epoch=400,
            eval_every=3,
        ),
    )
    return parser


def maybe_run_init_check(args, model, constraints, data) -> float:
    """Optionally print Stage D bounds and evaluate the initial test loss."""
    if not args.init_check:
        return float("nan")

    print("[init-check] constrained parameter bounds:")
    for name, constraint in constraints.items():
        print(f"[init-check] {name} init = {pretty_vector(constraint.init)}")
        print(f"[init-check] {name} lo   = {pretty_vector(constraint._lo)}")
        print(f"[init-check] {name} hi   = {pretty_vector(constraint._hi)}")

    forward_fn = build_constrained_forward(model, constraints)
    init_test_loss = rollout_loss_eval(
        model,
        data,
        args,
        forward_fn=forward_fn,
    )["test_traj_loss"]
    print(f"[init-check] initial test loss = {init_test_loss:.6e}")
    return float(init_test_loss)


def main() -> None:
    args = build_parser().parse_args()
    base_dir = os.path.dirname(__file__)
    run_dir, log_path = start_stage("stage_d", args.out_dir, sys.argv, base_dir=base_dir)

    stage_b = load_stage_json(args.input_b_json)
    stage_c = load_stage_json(args.input_c_json)
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

    model_params = clone_base_params()
    apply_body_state(model_params, stage_b)
    apply_wing_ls_state(model_params, stage_c)
    model = build_model(device=args.device, params=model_params)

    constraints = {
        "M_al": make_positive_diagonal_matrix_constraint(
            model.M_al,
            ratio=args.ratio_m_al,
            band=args.band,
            eps=args.band_eps,
            mode=args.constraint_mode,
        ),
        "I_al": make_positive_diagonal_matrix_constraint(
            model.I_al,
            ratio=args.ratio_i_al,
            band=args.band,
            eps=args.band_eps,
            mode=args.constraint_mode,
        ),
        "Theta_5": make_constraint(
            model.Theta_5,
            ratio=args.ratio_theta_5,
            band=args.theta_band or args.band,
            eps=args.band_eps,
            mode=args.constraint_mode,
        ),
        "Theta_6": make_constraint(
            model.Theta_6,
            ratio=args.ratio_theta_6,
            band=args.theta_band or args.band,
            eps=args.band_eps,
            mode=args.constraint_mode,
        ),
    }
    m_al_bound_overrides = _apply_positive_diagonal_bound_spec(
        constraints["M_al"],
        args.m_al_bound_spec,
        M_AL_DIAG_NAMES,
        eps=args.band_eps,
        label="M_al",
    )
    i_al_bound_overrides = _apply_positive_diagonal_bound_spec(
        constraints["I_al"],
        args.i_al_bound_spec,
        I_AL_DIAG_NAMES,
        eps=args.band_eps,
        label="I_al",
    )
    theta5_bound_overrides = _apply_elementwise_bound_spec(
        constraints["Theta_5"],
        args.theta5_bound_spec,
        THETA_5_NAMES,
        eps=args.band_eps,
        label="Theta_5",
    )
    theta6_bound_overrides = _apply_elementwise_bound_spec(
        constraints["Theta_6"],
        args.theta6_bound_spec,
        THETA_6_NAMES,
        eps=args.band_eps,
        label="Theta_6",
    )
    constrained_lrs = _stage_d_constrained_lrs(args)

    init_test_loss = maybe_run_init_check(args, model, constraints, data)
    stats = run_gd_training(
        model=model,
        data=data,
        args=args,
        constraint_kwargs={
            "constrain_M_al": constraints["M_al"],
            "constrain_I_al": constraints["I_al"],
            "constrain_Theta_5": constraints["Theta_5"],
            "constrain_Theta_6": constraints["Theta_6"],
        },
        restore_best_state=True,
        constrained_lrs=constrained_lrs,
    )

    test_rollout_plot_paths = []
    if args.plot_test_rollouts:
        test_rollout_plot_paths = plot_test_rollout_windows(
            model,
            data.test_traj,
            data.dt,
            args.plot_test_rollout_batch_size or args.batch_size_max,
            os.path.join(run_dir, "test_rollout_plots"),
            prefix="stage_d_test_rollout",
            batch_skip=args.batch_skip,
            num_windows=args.plot_test_rollout_windows,
            forward_fn=build_constrained_forward(model, constraints),
            integrator=args.integrator,
        )

    meta = build_gd_meta(args, data)
    meta.update(
        {
            "stage": "d",
            "log_path": log_path,
            "input_b_json": os.path.abspath(args.input_b_json),
            "input_c_json": os.path.abspath(args.input_c_json),
            "band": args.band,
            "theta_band": args.theta_band or args.band,
            "band_eps": args.band_eps,
            "constraint_mode": args.constraint_mode,
            "added_mass_inertia_constraint": "positive_diagonal",
            "restore_best_state": True,
            "test_rollout_plot_paths": test_rollout_plot_paths,
            "wing_model": "finite_angle_single_wing_rate_split_v1",
            "M_al_diag_order": ["Mxx", "Myy", "Mzz"],
            "I_al_diag_order": ["Ixx", "Iyy", "Izz"],
            "Theta_5_order": ["C_D0", "Delta_C_Dalpha", "C_Sbeta", "C_Lalpha"],
            "Theta_6_order": [
                "C_xbeta",
                "C_xp_d",
                "C_m0",
                "C_malpha",
                "C_mq_d",
                "C_zbeta",
                "C_zr_d",
            ],
            "ratio_m_al": args.ratio_m_al,
            "ratio_i_al": args.ratio_i_al,
            "ratio_theta_5": args.ratio_theta_5,
            "ratio_theta_6": args.ratio_theta_6,
            "lr_m_al": args.lr_m_al,
            "lr_i_al": args.lr_i_al,
            "lr_theta_5": args.lr_theta_5,
            "lr_theta_6": args.lr_theta_6,
            "constrained_lrs": constrained_lrs,
            "m_al_bound_spec": args.m_al_bound_spec,
            "i_al_bound_spec": args.i_al_bound_spec,
            "theta5_bound_spec": args.theta5_bound_spec,
            "theta6_bound_spec": args.theta6_bound_spec,
            "m_al_bound_overrides": m_al_bound_overrides,
            "i_al_bound_overrides": i_al_bound_overrides,
            "theta5_bound_overrides": theta5_bound_overrides,
            "theta6_bound_overrides": theta6_bound_overrides,
            "m_al_bounds_lower": constraints["M_al"]._lo,
            "m_al_bounds_upper": constraints["M_al"]._hi,
            "i_al_bounds_lower": constraints["I_al"]._lo,
            "i_al_bounds_upper": constraints["I_al"]._hi,
            "theta5_bounds_lower": constraints["Theta_5"]._lo,
            "theta5_bounds_upper": constraints["Theta_5"]._hi,
            "theta6_bounds_lower": constraints["Theta_6"]._lo,
            "theta6_bounds_upper": constraints["Theta_6"]._hi,
            "init_check": bool(args.init_check),
        }
    )
    if args.init_check and init_test_loss == init_test_loss:
        meta["init_test_loss"] = init_test_loss

    theta5_raw = constraints["Theta_5"].value().detach()
    theta6_raw = constraints["Theta_6"].value().detach()

    result = {
        "meta": meta,
        "wing_hydro_params": {
            # In stage-d, effective values are directly the constrained values.
            "Theta_5": theta5_raw,
            "Theta_6": theta6_raw,
            "Theta_5_raw": theta5_raw,
            "Theta_6_raw": theta6_raw,
        },
        "wing_added_mass_inertia": {
            "M_al": constraints["M_al"].value().detach(),
            "I_al": constraints["I_al"].value().detach(),
        },
        "stats": stats,
    }

    out_path = os.path.join(run_dir, "stage_d_result.json")
    save_json(out_path, result)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
