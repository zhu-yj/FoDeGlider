from __future__ import annotations


import argparse
import os
import sys
import torch

from SI_awug_crba_utils import (
    GdDefaults,
    apply_body_state,
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

BODY_DIAG_NAMES = ("x", "y", "z")
THETA_3_NAMES = ("X_u", "X_uu", "Y_v", "Y_vv", "Y_r", "Y_rr", "Z_w", "Z_ww", "Z_q", "Z_qq")
THETA_4_NAMES = ("K_p", "K_pp", "M_q", "M_qq", "M_w", "M_ww", "N_r", "N_rr", "N_v", "N_vv")
THETA_3_YZ_RAW_NAMES = ("X_u", "X_uu", "Z_w/Y_v", "Z_ww/Y_vv", "Z_q/-Y_r", "Z_qq/-Y_rr")
THETA_4_YZ_RAW_NAMES = ("K_p", "K_pp", "N_r/M_q", "N_rr/M_qq", "N_v/-M_w", "N_vv/-M_ww")


def _split_bound_spec(spec: str) -> list[str]:
    return [x.strip() for x in (spec or "").replace(";", ",").split(",") if x.strip()]


def _resolve_bound_index(key: str, names: tuple[str, ...], *, label: str) -> int:
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


def _make_bound_interval(parts: list[str], center: torch.Tensor, *, eps: float, label: str, entry: str):
    if len(parts) < 2:
        raise ValueError(f"{label}: bad bound entry {entry!r}")
    mode = parts[1].lower()
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
    elif mode in {"abs_symmetric", "symmetric"}:
        if len(parts) != 3:
            raise ValueError(f"{label}: abs_symmetric entry must be name:abs_symmetric:ratio, got {entry!r}")
        ratio = float(parts[2])
        mag = torch.maximum(torch.abs(center), torch.as_tensor(eps, dtype=center.dtype, device=center.device))
        half = ratio * mag
        lo = -half
        hi = half
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
    return mode, torch.minimum(lo, hi), torch.maximum(lo, hi)


def _record_bound_update(updates: list[dict], *, name: str, index: int, mode: str, init, lower, upper) -> None:
    updates.append(
        {
            "name": name,
            "index": int(index),
            "mode": mode,
            "init": float(init.detach().cpu()),
            "lower": float(lower.detach().cpu()),
            "upper": float(upper.detach().cpu()),
        }
    )


def _apply_vector_bound_spec(constraint, spec: str, names: tuple[str, ...], *, eps: float, label: str) -> list[dict]:
    """Override selected unconstrained tensor bounds, matching Stage D syntax."""
    entries = _split_bound_spec(spec)
    if not entries:
        return []
    init = constraint.init.reshape(-1)
    lo_flat = constraint._lo.reshape(-1)
    hi_flat = constraint._hi.reshape(-1)
    raw_flat = constraint.raw.reshape(-1)
    updates: list[dict] = []
    with torch.no_grad():
        for entry in entries:
            parts = [p.strip() for p in entry.split(":")]
            idx = _resolve_bound_index(parts[0], names, label=label)
            mode, lo, hi = _make_bound_interval(parts, init[idx], eps=eps, label=label, entry=entry)
            lo_flat[idx].copy_(lo)
            hi_flat[idx].copy_(hi)
            raw_flat[idx].copy_(_clamp_tensor(raw_flat[idx], lo, hi))
            _record_bound_update(updates, name=names[idx], index=idx, mode=mode, init=init[idx], lower=lo, upper=hi)
    return updates


def _apply_positive_diag_bound_spec(constraint, spec: str, *, eps: float, label: str) -> list[dict]:
    """Override selected positive diagonal bounds for M_ab/I_ab."""
    entries = _split_bound_spec(spec)
    if not entries:
        return []
    init_diag = constraint.init.diagonal().detach().clone()
    updates: list[dict] = []
    updated_raw: set[int] = set()
    floor = torch.as_tensor(1e-12, dtype=init_diag.dtype, device=init_diag.device)

    if hasattr(constraint, "_lo_raw") and constraint.raw.numel() == 2:
        raw_names = ("x", "yz")
        eff_to_raw = (0, 1, 1)
        raw_lo = constraint._lo_raw
        raw_hi = constraint._hi_raw
        raw_flat = constraint.raw.reshape(-1)
    else:
        raw_names = BODY_DIAG_NAMES
        eff_to_raw = (0, 1, 2)
        raw_lo = constraint._lo_diag
        raw_hi = constraint._hi_diag
        raw_flat = constraint.raw.reshape(-1)

    with torch.no_grad():
        for entry in entries:
            parts = [p.strip() for p in entry.split(":")]
            idx = _resolve_bound_index(parts[0], BODY_DIAG_NAMES, label=label)
            mode, lo, hi = _make_bound_interval(parts, init_diag[idx], eps=eps, label=label, entry=entry)
            lo = torch.maximum(lo, floor)
            hi = torch.maximum(hi, lo)
            raw_idx = eff_to_raw[idx]
            if raw_idx in updated_raw:
                lo = torch.maximum(raw_lo[raw_idx], lo)
                hi = torch.minimum(raw_hi[raw_idx], hi)
                if bool((lo > hi).item()):
                    raise ValueError(f"{label}: conflicting y/z diagonal bounds after {entry!r}")
            else:
                updated_raw.add(raw_idx)
            raw_lo[raw_idx].copy_(lo)
            raw_hi[raw_idx].copy_(hi)
            raw_flat[raw_idx].copy_(_clamp_tensor(raw_flat[raw_idx], lo, hi))
            _record_bound_update(
                updates,
                name=BODY_DIAG_NAMES[idx],
                index=idx,
                mode=mode,
                init=init_diag[idx],
                lower=lo,
                upper=hi,
            )

        if hasattr(constraint, "_lo_raw") and constraint.raw.numel() == 2:
            constraint._lo_diag = torch.stack([raw_lo[0], raw_lo[1], raw_lo[1]]).detach().clone()
            constraint._hi_diag = torch.stack([raw_hi[0], raw_hi[1], raw_hi[1]]).detach().clone()
            constraint._lo = torch.diag(constraint._lo_diag).clone()
            constraint._hi = torch.diag(constraint._hi_diag).clone()
        else:
            constraint._lo = torch.diag(raw_lo).clone()
            constraint._hi = torch.diag(raw_hi).clone()
    return updates


def _theta_effective_to_yz_raw(idx: int) -> tuple[int, float]:
    mapping = (
        (0, 1.0),
        (1, 1.0),
        (2, 1.0),
        (3, 1.0),
        (4, -1.0),
        (5, -1.0),
        (2, 1.0),
        (3, 1.0),
        (4, 1.0),
        (5, 1.0),
    )
    return mapping[idx]


def _refresh_yz_theta_bounds(constraint) -> None:
    lo = constraint._lo_raw
    hi = constraint._hi_raw
    lo_full = torch.stack([lo[0], lo[1], lo[2], lo[3], -hi[4], -hi[5], lo[2], lo[3], lo[4], lo[5]])
    hi_full = torch.stack([hi[0], hi[1], hi[2], hi[3], -lo[4], -lo[5], hi[2], hi[3], hi[4], hi[5]])
    constraint._lo = lo_full.reshape_as(constraint.init).clone()
    constraint._hi = hi_full.reshape_as(constraint.init).clone()


def _apply_body_theta_bound_spec(
    constraint,
    spec: str,
    names: tuple[str, ...],
    *,
    eps: float,
    label: str,
) -> list[dict]:
    """Override Theta_3/4 bounds, supporting Stage-B y/z symmetric constraints."""
    entries = _split_bound_spec(spec)
    if not entries:
        return []

    if not (hasattr(constraint, "_lo_raw") and constraint.raw.numel() == 6 and constraint.init.numel() == 10):
        return _apply_vector_bound_spec(constraint, spec, names, eps=eps, label=label)

    init_eff = constraint.init.reshape(-1)
    raw_lo = constraint._lo_raw
    raw_hi = constraint._hi_raw
    raw_flat = constraint.raw.reshape(-1)
    updates: list[dict] = []
    updated_raw: set[int] = set()
    with torch.no_grad():
        for entry in entries:
            parts = [p.strip() for p in entry.split(":")]
            idx = _resolve_bound_index(parts[0], names, label=label)
            mode, lo_eff, hi_eff = _make_bound_interval(parts, init_eff[idx], eps=eps, label=label, entry=entry)
            raw_idx, sign = _theta_effective_to_yz_raw(idx)
            if sign < 0.0:
                lo_raw = -hi_eff
                hi_raw = -lo_eff
            else:
                lo_raw = lo_eff
                hi_raw = hi_eff
            lo_raw = torch.minimum(lo_raw, hi_raw)
            hi_raw = torch.maximum(lo_raw, hi_raw)
            if raw_idx in updated_raw:
                lo_raw = torch.maximum(raw_lo[raw_idx], lo_raw)
                hi_raw = torch.minimum(raw_hi[raw_idx], hi_raw)
                if bool((lo_raw > hi_raw).item()):
                    raise ValueError(f"{label}: conflicting symmetric pair bounds after {entry!r}")
            else:
                updated_raw.add(raw_idx)
            raw_lo[raw_idx].copy_(lo_raw)
            raw_hi[raw_idx].copy_(hi_raw)
            raw_flat[raw_idx].copy_(_clamp_tensor(raw_flat[raw_idx], lo_raw, hi_raw))
            _record_bound_update(
                updates,
                name=names[idx],
                index=idx,
                mode=mode,
                init=init_eff[idx],
                lower=lo_eff,
                upper=hi_eff,
            )
        _refresh_yz_theta_bounds(constraint)
    return updates

def build_parser() -> argparse.ArgumentParser:
    """Build the Stage B command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Stage B constrained gradient descent for body inertia and body hydrodynamic coefficients."
            " / Constrained GD for body inertia and body hydrodynamic coefficients."
        )
    )
    parser.add_argument("--input-a-json", required=True, help="Path to stage_a_result.json.")
    parser.add_argument("--train-files", required=True)
    parser.add_argument("--test-files", required=True)
    parser.add_argument("--out-dir", default="log/si_awug_crba")

    parser.add_argument("--ratio-m-ab", type=float, default=1.0)
    parser.add_argument("--ratio-i-ab", type=float, default=1.0)
    parser.add_argument("--ratio-theta-3", type=float, default=2.0)
    parser.add_argument("--ratio-theta-4", type=float, default=2.0)
    parser.add_argument(
        "--m-ab-bound-spec",
        default="",
        help=(
            "Per-axis body added-mass bound overrides. Entries are separated by commas or semicolons. "
            "Formats: x:relative:ratio, y:abs:lower:upper, z:abs_scaled:ratio, x:fixed[:value]. "
            "Names: x, y, z. With --train-body-yz-symmetry, y and z share one feasible interval."
        ),
    )
    parser.add_argument(
        "--i-ab-bound-spec",
        default="",
        help=(
            "Per-axis body added-inertia bound overrides. Entries are separated by commas or semicolons. "
            "Formats: x:relative:ratio, y:abs:lower:upper, z:abs_scaled:ratio, x:fixed[:value]. "
            "Names: x, y, z. With --train-body-yz-symmetry, y and z share one feasible interval."
        ),
    )
    parser.add_argument(
        "--theta3-bound-spec",
        default="",
        help=(
            "Per-coefficient Theta_3 bound overrides. Entries are separated by commas or semicolons. "
            "Formats: name:relative:ratio, name:abs:lower:upper, name:abs_scaled:ratio, "
            "name:abs_symmetric:ratio, name:fixed[:value]. "
            "Names: X_u, X_uu, Y_v, Y_vv, Y_r, Y_rr, Z_w, Z_ww, Z_q, Z_qq. "
            "With --train-body-yz-symmetry, paired y/z entries are intersected in the shared raw space."
        ),
    )
    parser.add_argument(
        "--theta4-bound-spec",
        default="",
        help=(
            "Per-coefficient Theta_4 bound overrides. Entries are separated by commas or semicolons. "
            "Formats: name:relative:ratio, name:abs:lower:upper, name:abs_scaled:ratio, "
            "name:abs_symmetric:ratio, name:fixed[:value]. "
            "Names: K_p, K_pp, M_q, M_qq, M_w, M_ww, N_r, N_rr, N_v, N_vv. "
            "With --train-body-yz-symmetry, paired y/z entries are intersected in the shared raw space."
        ),
    )
    parser.add_argument("--band", choices=["relative", "abs_scaled"], default="abs_scaled")
    parser.add_argument(
        "--theta-band",
        choices=["relative", "abs_scaled", "abs_symmetric"],
        default=None,
        help=(
            "Optional bound mode for Theta_3/Theta_4 only. "
            "abs_symmetric means [-ratio*abs(init), +ratio*abs(init)]."
        ),
    )
    parser.add_argument("--band-eps", type=float, default=1e-2)
    parser.add_argument(
        "--constraint-mode",
        choices=["project"],
        default="project",
        help="Use projected box constraints for all bounded Stage B parameters.",
    )
    parser.add_argument(
        "--post-train-body-yz-symmetry",
        action="store_true",
        help=(
            "After Stage B training/restoring best state, overwrite body y-direction "
            "added-mass/inertia and Theta entries from the z-direction values using "
            "the Stage-A body symmetry signs."
        ),
    )
    parser.add_argument(
        "--train-body-yz-symmetry",
        action="store_true",
        help=(
            "Enforce body y/z symmetry during Stage B training by sharing the z-direction "
            "source parameters in the constrained forward path."
        ),
    )
    parser.add_argument("--init-check", action="store_true")

    add_gd_arguments(
        parser,
        defaults=GdDefaults(
            epochs=30,
            lr=1e-4,
            grad_clip=1.0,
            dt_base=1.0 / 90.0,
            batch_size_min=5,
            batch_size_max=30,
            batch_size_schedule="linear",
            batch_skip=1,
            norm_mode="zscore",
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


def _flat_list(x):
    return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]


def _clamp_tensor(x, lo, hi):
    return torch.maximum(torch.minimum(x, hi), lo)


def _intersect_scalar_bounds(lo_a, hi_a, lo_b, hi_b, name: str):
    lo = torch.maximum(lo_a, lo_b)
    hi = torch.minimum(hi_a, hi_b)
    if bool((lo > hi).item()):
        raise ValueError(
            f"{name} has no feasible shared y/z interval: "
            f"[{float(lo_a):.6g}, {float(hi_a):.6g}] vs [{float(lo_b):.6g}, {float(hi_b):.6g}]"
        )
    return lo, hi


class BodyYZSymmetricDiagonalConstraint:
    """Positive diagonal matrix with y and z sharing the z-source parameter."""

    def __init__(self, parameter, *, ratio: float, band: str, eps: float, mode: str = "project"):
        if mode != "project":
            raise ValueError("--train-body-yz-symmetry requires project constraints")
        base = make_positive_diagonal_matrix_constraint(
            parameter,
            ratio=ratio,
            band=band,
            eps=eps,
            mode=mode,
        )
        init_diag = base.init.diagonal().detach().clone()
        lo_diag = base._lo_diag.detach().clone()
        hi_diag = base._hi_diag.detach().clone()
        lo_shared, hi_shared = _intersect_scalar_bounds(
            lo_diag[1], hi_diag[1], lo_diag[2], hi_diag[2], "body diagonal y/z"
        )
        self.mode = mode
        self.raw = torch.nn.Parameter(
            torch.stack(
                [
                    _clamp_tensor(init_diag[0], lo_diag[0], hi_diag[0]),
                    _clamp_tensor(init_diag[2], lo_shared, hi_shared),
                ]
            )
        )
        self._lo_raw = torch.stack([lo_diag[0], lo_shared]).detach().clone()
        self._hi_raw = torch.stack([hi_diag[0], hi_shared]).detach().clone()
        init_full = torch.diag(torch.stack([self.raw[0].detach(), self.raw[1].detach(), self.raw[1].detach()]))
        self.init = init_full.clone()
        self._lo_diag = torch.stack([lo_diag[0], lo_shared, lo_shared]).detach().clone()
        self._hi_diag = torch.stack([hi_diag[0], hi_shared, hi_shared]).detach().clone()
        self._lo = torch.diag(self._lo_diag).clone()
        self._hi = torch.diag(self._hi_diag).clone()

    def value(self):
        return torch.diag(torch.stack([self.raw[0], self.raw[1], self.raw[1]]))

    def project_(self) -> None:
        with torch.no_grad():
            self.raw.copy_(_clamp_tensor(self.raw, self._lo_raw, self._hi_raw))


class BodyYZSymmetricThetaConstraint:
    """Theta vector constrained to the Stage-A y/z body symmetry during training."""

    def __init__(self, parameter, *, ratio: float, band: str, eps: float, mode: str = "project"):
        if mode != "project":
            raise ValueError("--train-body-yz-symmetry requires project constraints")
        base = make_constraint(parameter, ratio=ratio, band=band, eps=eps, mode=mode)
        init = base.init.detach().clone().reshape(-1)
        lo = base._lo.detach().clone().reshape(-1)
        hi = base._hi.detach().clone().reshape(-1)
        if init.numel() != 10:
            raise ValueError("BodyYZSymmetricThetaConstraint expects a 10-element Theta vector")

        lo_2, hi_2 = _intersect_scalar_bounds(lo[2], hi[2], lo[6], hi[6], "theta y/z pair 2-6")
        lo_3, hi_3 = _intersect_scalar_bounds(lo[3], hi[3], lo[7], hi[7], "theta y/z pair 3-7")
        lo_4, hi_4 = _intersect_scalar_bounds(-hi[4], -lo[4], lo[8], hi[8], "theta opposite-sign pair 4-8")
        lo_5, hi_5 = _intersect_scalar_bounds(-hi[5], -lo[5], lo[9], hi[9], "theta opposite-sign pair 5-9")

        raw_init = torch.stack(
            [
                _clamp_tensor(init[0], lo[0], hi[0]),
                _clamp_tensor(init[1], lo[1], hi[1]),
                _clamp_tensor(init[6], lo_2, hi_2),
                _clamp_tensor(init[7], lo_3, hi_3),
                _clamp_tensor(init[8], lo_4, hi_4),
                _clamp_tensor(init[9], lo_5, hi_5),
            ]
        )
        self.mode = mode
        self.raw = torch.nn.Parameter(raw_init)
        self._lo_raw = torch.stack([lo[0], lo[1], lo_2, lo_3, lo_4, lo_5]).detach().clone()
        self._hi_raw = torch.stack([hi[0], hi[1], hi_2, hi_3, hi_4, hi_5]).detach().clone()
        self.init = self._expand(self.raw.detach()).reshape_as(base.init).clone()
        lo_full = torch.stack([lo[0], lo[1], lo_2, lo_3, -hi_4, -hi_5, lo_2, lo_3, lo_4, lo_5])
        hi_full = torch.stack([hi[0], hi[1], hi_2, hi_3, -lo_4, -lo_5, hi_2, hi_3, hi_4, hi_5])
        self._lo = lo_full.reshape_as(base.init).clone()
        self._hi = hi_full.reshape_as(base.init).clone()

    @staticmethod
    def _expand(raw):
        return torch.stack([raw[0], raw[1], raw[2], raw[3], -raw[4], -raw[5], raw[2], raw[3], raw[4], raw[5]])

    def value(self):
        return self._expand(self.raw).reshape_as(self.init)

    def project_(self) -> None:
        with torch.no_grad():
            self.raw.copy_(_clamp_tensor(self.raw, self._lo_raw, self._hi_raw))


def make_body_yz_symmetric_diagonal_constraint(parameter, *, ratio: float, band: str, eps: float, mode: str):
    return BodyYZSymmetricDiagonalConstraint(parameter, ratio=ratio, band=band, eps=eps, mode=mode)


def make_body_yz_symmetric_theta_constraint(parameter, *, ratio: float, band: str, eps: float, mode: str):
    return BodyYZSymmetricThetaConstraint(parameter, ratio=ratio, band=band, eps=eps, mode=mode)


def apply_post_train_body_yz_symmetry(constraints) -> dict:
    """Force Stage-B body y/z symmetry after training, using z as source."""
    summary = {}
    for name in ("M_ab", "I_ab"):
        constraint = constraints[name]
        before = _flat_list(constraint.value().diagonal())
        with torch.no_grad():
            if getattr(constraint, "mode", None) != "project":
                raise ValueError("--post-train-body-yz-symmetry requires project constraints")
            constraint.raw[1].copy_(constraint.raw[2])
            constraint.project_()
        after = _flat_list(constraint.value().diagonal())
        summary[name] = {"before_diag": before, "after_diag": after, "rule": "diag_y=diag_z"}

    theta_rule = "theta[2]=theta[6], theta[3]=theta[7], theta[4]=-theta[8], theta[5]=-theta[9]"
    for name in ("Theta_3", "Theta_4"):
        constraint = constraints[name]
        before = _flat_list(constraint.value())
        with torch.no_grad():
            if getattr(constraint, "mode", None) != "project":
                raise ValueError("--post-train-body-yz-symmetry requires project constraints")
            flat = constraint.raw.reshape(-1)
            flat[2].copy_(flat[6])
            flat[3].copy_(flat[7])
            flat[4].copy_(-flat[8])
            flat[5].copy_(-flat[9])
            constraint.project_()
        after = _flat_list(constraint.value())
        summary[name] = {"before": before, "after": after, "rule": theta_rule}
    return summary


def maybe_run_init_check(args, model, constraints, data) -> float:
    """Optionally print constraint bounds and evaluate the initial test loss."""
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
    run_dir, log_path = start_stage("stage_b", args.out_dir, sys.argv, base_dir=base_dir)

    stage_a = load_stage_json(args.input_a_json)
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

    model_params = apply_body_state(clone_base_params(), stage_a)
    model = build_model(device=args.device, params=model_params)

    if args.train_body_yz_symmetry:
        diag_constraint_fn = make_body_yz_symmetric_diagonal_constraint
        theta_constraint_fn = make_body_yz_symmetric_theta_constraint
    else:
        diag_constraint_fn = make_positive_diagonal_matrix_constraint
        theta_constraint_fn = make_constraint

    constraints = {
        "M_ab": diag_constraint_fn(
            model.M_ab,
            ratio=args.ratio_m_ab,
            band=args.band,
            eps=args.band_eps,
            mode=args.constraint_mode,
        ),
        "I_ab": diag_constraint_fn(
            model.I_ab,
            ratio=args.ratio_i_ab,
            band=args.band,
            eps=args.band_eps,
            mode=args.constraint_mode,
        ),
        "Theta_3": theta_constraint_fn(
            model.Theta_3,
            ratio=args.ratio_theta_3,
            band=args.theta_band or args.band,
            eps=args.band_eps,
            mode=args.constraint_mode,
        ),
        "Theta_4": theta_constraint_fn(
            model.Theta_4,
            ratio=args.ratio_theta_4,
            band=args.theta_band or args.band,
            eps=args.band_eps,
            mode=args.constraint_mode,
        ),
    }

    m_ab_bound_overrides = _apply_positive_diag_bound_spec(
        constraints["M_ab"],
        args.m_ab_bound_spec,
        eps=args.band_eps,
        label="M_ab",
    )
    i_ab_bound_overrides = _apply_positive_diag_bound_spec(
        constraints["I_ab"],
        args.i_ab_bound_spec,
        eps=args.band_eps,
        label="I_ab",
    )
    theta3_bound_overrides = _apply_body_theta_bound_spec(
        constraints["Theta_3"],
        args.theta3_bound_spec,
        THETA_3_NAMES,
        eps=args.band_eps,
        label="Theta_3",
    )
    theta4_bound_overrides = _apply_body_theta_bound_spec(
        constraints["Theta_4"],
        args.theta4_bound_spec,
        THETA_4_NAMES,
        eps=args.band_eps,
        label="Theta_4",
    )

    init_test_loss = maybe_run_init_check(args, model, constraints, data)
    stats = run_gd_training(
        model=model,
        data=data,
        args=args,
        constraint_kwargs={
            "constrain_M_ab": constraints["M_ab"],
            "constrain_I_ab": constraints["I_ab"],
            "constrain_Theta_3": constraints["Theta_3"],
            "constrain_Theta_4": constraints["Theta_4"],
        },
        restore_best_state=True,
    )

    post_train_body_yz_symmetry = None
    if args.post_train_body_yz_symmetry and args.train_body_yz_symmetry:
        post_train_body_yz_symmetry = {"already_enforced_during_training": True}
    elif args.post_train_body_yz_symmetry:
        print("[post-train-yz-sym] applying Stage-A body y/z symmetry using z as source", flush=True)
        post_train_body_yz_symmetry = apply_post_train_body_yz_symmetry(constraints)
        post_loss = rollout_loss_eval(
            model,
            data,
            args,
            forward_fn=build_constrained_forward(model, constraints),
        )["test_traj_loss"]
        stats["post_train_body_yz_symmetry_test_loss"] = float(post_loss)
        print(f"[post-train-yz-sym] test_loss_after_projection = {post_loss:.6e}", flush=True)

    test_rollout_plot_paths = []
    if args.plot_test_rollouts:
        test_rollout_plot_paths = plot_test_rollout_windows(
            model,
            data.test_traj,
            data.dt,
            args.plot_test_rollout_batch_size or args.batch_size_max,
            os.path.join(run_dir, "test_rollout_plots"),
            prefix="stage_b_test_rollout",
            batch_skip=args.batch_skip,
            num_windows=args.plot_test_rollout_windows,
            forward_fn=build_constrained_forward(model, constraints),
        )

    meta = build_gd_meta(args, data)
    meta.update(
        {
            "stage": "b",
            "log_path": log_path,
            "input_a_json": os.path.abspath(args.input_a_json),
            "band": args.band,
            "theta_band": args.theta_band or args.band,
            "band_eps": args.band_eps,
            "constraint_mode": args.constraint_mode,
            "added_mass_inertia_constraint": (
                "positive_diagonal_body_yz_symmetric"
                if args.train_body_yz_symmetry
                else "positive_diagonal"
            ),
            "theta_constraint": (
                "body_yz_symmetric"
                if args.train_body_yz_symmetry
                else "box"
            ),
            "restore_best_state": True,
            "train_body_yz_symmetry": bool(args.train_body_yz_symmetry),
            "post_train_body_yz_symmetry": bool(args.post_train_body_yz_symmetry),
            "post_train_body_yz_symmetry_summary": post_train_body_yz_symmetry,
            "test_rollout_plot_paths": test_rollout_plot_paths,
            "M_ab_order": list(BODY_DIAG_NAMES),
            "I_ab_order": list(BODY_DIAG_NAMES),
            "Theta_3_order": list(THETA_3_NAMES),
            "Theta_4_order": list(THETA_4_NAMES),
            "Theta_3_yz_raw_order": list(THETA_3_YZ_RAW_NAMES),
            "Theta_4_yz_raw_order": list(THETA_4_YZ_RAW_NAMES),
            "ratio_m_ab": args.ratio_m_ab,
            "ratio_i_ab": args.ratio_i_ab,
            "ratio_theta_3": args.ratio_theta_3,
            "ratio_theta_4": args.ratio_theta_4,
            "m_ab_bound_spec": args.m_ab_bound_spec,
            "i_ab_bound_spec": args.i_ab_bound_spec,
            "theta3_bound_spec": args.theta3_bound_spec,
            "theta4_bound_spec": args.theta4_bound_spec,
            "m_ab_bound_overrides": m_ab_bound_overrides,
            "i_ab_bound_overrides": i_ab_bound_overrides,
            "theta3_bound_overrides": theta3_bound_overrides,
            "theta4_bound_overrides": theta4_bound_overrides,
            "m_ab_bounds_lower": constraints["M_ab"]._lo,
            "m_ab_bounds_upper": constraints["M_ab"]._hi,
            "i_ab_bounds_lower": constraints["I_ab"]._lo,
            "i_ab_bounds_upper": constraints["I_ab"]._hi,
            "theta3_bounds_lower": constraints["Theta_3"]._lo,
            "theta3_bounds_upper": constraints["Theta_3"]._hi,
            "theta4_bounds_lower": constraints["Theta_4"]._lo,
            "theta4_bounds_upper": constraints["Theta_4"]._hi,
            "init_check": bool(args.init_check),
        }
    )
    if args.init_check and init_test_loss == init_test_loss:
        meta["init_test_loss"] = init_test_loss

    theta3_raw = constraints["Theta_3"].value().detach()
    theta4_raw = constraints["Theta_4"].value().detach()
    theta3_eff = model._apply_body_symmetry(theta3_raw)
    theta4_eff = model._apply_body_symmetry(theta4_raw)

    result = {
        "meta": meta,
        "body_hydro_params": {
            # Save effective vectors used in forward dynamics to avoid ambiguity.
            "Theta_3": theta3_eff,
            "Theta_4": theta4_eff,
            # Keep raw constrained vectors for debugging/reproducibility.
            "Theta_3_raw": theta3_raw,
            "Theta_4_raw": theta4_raw,
        },
        "body_added_mass_inertia": {
            "M_ab": constraints["M_ab"].value().detach(),
            "I_ab": constraints["I_ab"].value().detach(),
        },
        "stats": stats,
    }

    out_path = os.path.join(run_dir, "stage_b_result.json")
    save_json(out_path, result)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
