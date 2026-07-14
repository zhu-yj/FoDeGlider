"""
Stage E entry for positive quadratic Bernstein scaling gates.

The upstream Stage B/D parameters and the shared GD loop are unchanged.  This
entry replaces only the sweep-dependent gates by constrained quadratic
Bernstein functions of the effective geometry ratio eta.

Hydrodynamic coefficient gate:
    K_i(eta) = beta_0 (1-eta)^2 + 2 beta_1 eta (1-eta) + eta^2
    beta_0 > 0, beta_1 > 0, K_i(1) = 1.

Added-mass / added-inertia gate:
    K_A,i(eta) = 2 gamma eta (1-eta) + eta^2
    gamma > 0, K_A,i(0) = 0, K_A,i(1) = 1.

Bernstein control values are trained directly and projected after each
optimizer step to eps <= beta <= upper.  Accordingly, an added-mass gate is
zero at its required eta=0 endpoint and strictly positive for eta in (0, 1].
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from SI_awug_crba_utils import (
    AWUGModelFromTex,
    GdDefaults,
    add_gd_arguments,
    apply_body_state,
    apply_wing_gd_state,
    build_gd_meta,
    build_named_parameter_lr_groups,
    clone_base_params,
    eval_windows_rmse,
    load_stage_data,
    load_stage_json,
    plot_convergence,
    plot_test_rollout_windows,
    plot_window_rmse_boxplot,
    rollout_loss_eval,
    run_gd_training,
    save_json,
    save_window_eval,
    start_stage,
)


class PositiveQuadraticBernsteinModel(AWUGModelFromTex):
    """AWUG model with positive quadratic Bernstein sweep gates."""

    POSITIVE_EPS = 1e-8
    DEFAULT_GATE_UPPER = 20.0
    ADDED_GATE_NAMES = ("k_XA_x", "k_XA_y", "k_XA_z", "k_KA", "k_MA", "k_NA")
    UNUSED_ADDED_GATE_NAMES = (
        "k_XA_x2",
        "k_XA_y2",
        "k_XA_z2",
        "k_KA2",
        "k_MA2",
        "k_NA2",
    )
    HYDRO_X5_CHANNELS = {
        "C_D0": ("C_D0_k1", "C_D0_k2"),
        "Delta_C_Dalpha": ("C_Da2_k1", "C_Da2_k2"),
        "C_Sbeta": ("C_Yb_k1", "C_Yb_k2"),
        "C_Lalpha": ("C_La_k1", "C_La_k2"),
    }
    HYDRO_X6_CHANNELS = {
        "C_xbeta": ("C_xbeta_k1", "C_xbeta_k2"),
        "C_xp_d": ("C_xp_d_k1", "C_xp_d_k2"),
        "C_m0": ("C_m0_k1", "C_m0_k2"),
        "C_malpha": ("C_malpha_k1", "C_malpha_k2"),
        "C_mq_d": ("C_mq_d_k1", "C_mq_d_k2"),
        "C_zbeta": ("C_zbeta_k1", "C_zbeta_k2"),
        "C_zr_d": ("C_zr_d_k1", "C_zr_d_k2"),
    }

    @classmethod
    def _validate_gate_upper(cls, value: float | None, *, name: str) -> float:
        upper = cls.DEFAULT_GATE_UPPER if value is None else float(value)
        if upper <= cls.POSITIVE_EPS:
            raise ValueError(f"{name} must be greater than {cls.POSITIVE_EPS:g}, got {upper:g}")
        return upper

    def __init__(
        self,
        *args,
        hydro_gate_upper: float | None = None,
        added_gate_upper: float | None = None,
        **kwargs,
    ) -> None:
        # The inherited k2 parameters are used as beta_1 controls for hydro
        # gates; therefore the legacy freeze_k2 option must not be applied.
        self.hydro_gate_upper = self._validate_gate_upper(hydro_gate_upper, name="hydro_gate_upper")
        self.added_gate_upper = self._validate_gate_upper(added_gate_upper, name="added_gate_upper")
        kwargs.pop("freeze_k2", None)
        super().__init__(*args, freeze_k2=False, **kwargs)
        self._initialize_bernstein_baseline()

    def _initialize_bernstein_baseline(self) -> None:
        """Initialize K_i=1 and K_A,i=eta without training useless controls."""
        with torch.no_grad():
            for name in self.ADDED_GATE_NAMES:
                getattr(self, name).fill_(0.5)
            for params in (self.k_g2_X5, self.k_g2_X6):
                for control in params.values():
                    control.fill_(1.0)

        # A quadratic endpoint-fixed added-mass gate has only gamma as a
        # free control.  Legacy second-order correction parameters are unused.
        for name in self.UNUSED_ADDED_GATE_NAMES:
            control = getattr(self, name)
            control.requires_grad_(False)
            control.data.zero_()
        self.project_controls_()

    @torch.no_grad()
    def project_controls_(self) -> None:
        """Project trainable Bernstein controls into positive bounded boxes."""
        for name in self.ADDED_GATE_NAMES:
            getattr(self, name).clamp_(min=self.POSITIVE_EPS, max=self.added_gate_upper)
        for params in (self.k_g2_X5, self.k_g2_X6):
            for control in params.values():
                control.clamp_(min=self.POSITIVE_EPS, max=self.hydro_gate_upper)

    def _gate_sweep(
        self,
        eta: torch.Tensor,
        beta_0: torch.Tensor,
        beta_1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Positive hydrodynamic scaling with fixed endpoint K_i(1)=1."""
        if beta_1 is None:
            raise ValueError("Quadratic Bernstein hydrodynamic gate requires two controls.")
        one_minus_eta = 1.0 - eta
        return beta_0 * one_minus_eta**2 + 2.0 * beta_1 * eta * one_minus_eta + eta**2

    def _gate_added_mass(
        self,
        eta: torch.Tensor,
        gamma: torch.Tensor,
        unused: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Endpoint-fixed added-mass scaling, positive away from eta=0."""
        del unused
        return 2.0 * gamma * eta * (1.0 - eta) + eta**2


BERNSTEIN_GATE_LR_GROUP_PARAMS = {
    "added_mass_gates": tuple(name for name in PositiveQuadraticBernsteinModel.ADDED_GATE_NAMES if name.startswith("k_XA")),
    "added_inertia_gates": tuple(
        name for name in PositiveQuadraticBernsteinModel.ADDED_GATE_NAMES if name.startswith(("k_KA", "k_MA", "k_NA"))
    ),
    "force_gates": tuple(
        f"k_g2_X5.{param_name}"
        for pair in PositiveQuadraticBernsteinModel.HYDRO_X5_CHANNELS.values()
        for param_name in pair
    ),
    "torque_gates": tuple(
        f"k_g2_X6.{param_name}"
        for pair in PositiveQuadraticBernsteinModel.HYDRO_X6_CHANNELS.values()
        for param_name in pair
    ),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the constrained Bernstein Stage E command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Stage E: train positive quadratic Bernstein gates for sweep-dependent "
            "hydrodynamic and added-mass/inertia scaling."
        )
    )
    parser.add_argument("--input-b-json", required=True, help="Path to stage_b_result.json.")
    parser.add_argument("--input-d-json", required=True, help="Path to stage_d_result.json.")
    parser.add_argument("--train-files", required=True)
    parser.add_argument("--test-files", required=True)
    parser.add_argument("--out-dir", default="log/si_awug_crba")
    parser.add_argument(
        "--disable-added-mass-gates",
        action="store_true",
        help="Only train hydrodynamic Bernstein controls.",
    )
    parser.add_argument(
        "--gate-upper",
        type=float,
        default=PositiveQuadraticBernsteinModel.DEFAULT_GATE_UPPER,
        help="Shared upper bound for all trainable Bernstein control values.",
    )
    parser.add_argument(
        "--hydro-gate-upper",
        type=float,
        default=None,
        help="Optional upper bound override for hydrodynamic Bernstein controls.",
    )
    parser.add_argument(
        "--added-gate-upper",
        type=float,
        default=None,
        help="Optional upper bound override for added-mass/inertia Bernstein controls.",
    )
    parser.add_argument(
        "--lr-added-mass-gates",
        type=float,
        default=None,
        help="Optional Adam lr override for added-mass Bernstein controls.",
    )
    parser.add_argument(
        "--lr-added-inertia-gates",
        type=float,
        default=None,
        help="Optional Adam lr override for added-inertia Bernstein controls.",
    )
    parser.add_argument(
        "--lr-force-gates",
        type=float,
        default=None,
        help="Optional Adam lr override for hydrodynamic force Bernstein controls.",
    )
    parser.add_argument(
        "--lr-torque-gates",
        type=float,
        default=None,
        help="Optional Adam lr override for hydrodynamic moment/torque Bernstein controls.",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--eval-windows", action="store_true")
    parser.add_argument("--init-check", action="store_true")
    add_gd_arguments(
        parser,
        defaults=GdDefaults(
            epochs=20,
            lr=1e-3,
            grad_clip=1.0,
            dt_base=1.0 / 90.0,
            batch_size_min=8,
            batch_size_max=64,
            batch_size_schedule="linear",
            batch_skip=1,
            norm_mode="none",
            norm_stats_json=None,
            loss_weight_mode="none",
            variance_weights_json=None,
            task_weights_spec=None,
            train_progress_every=30,
            max_windows_per_epoch=400,
            eval_every=3,
        ),
    )
    return parser


def stage_e_named_parameter_lrs(args) -> dict:
    """Collect optional per-gate optimizer LR overrides for Bernstein Stage E."""
    return build_named_parameter_lr_groups(
        BERNSTEIN_GATE_LR_GROUP_PARAMS,
        {
            "added_mass_gates": args.lr_added_mass_gates,
            "added_inertia_gates": args.lr_added_inertia_gates,
            "force_gates": args.lr_force_gates,
            "torque_gates": args.lr_torque_gates,
        },
    )


def gate_lr_meta(args) -> dict:
    return {
        "added_mass_gates": args.lr_added_mass_gates,
        "added_inertia_gates": args.lr_added_inertia_gates,
        "force_gates": args.lr_force_gates,
        "torque_gates": args.lr_torque_gates,
    }


def build_bernstein_model(
    *,
    device: str,
    params: dict,
    train_added: bool,
    train_hydro: bool,
    hydro_gate_upper: float | None = None,
    added_gate_upper: float | None = None,
):
    """Build a model whose only trainable Stage E values are direct controls."""
    return PositiveQuadraticBernsteinModel(
        params=params,
        device=device,
        train_wing_added_mass_inertia_gates=train_added,
        train_wing_hydro_gates=train_hydro,
        hydro_gate_upper=hydro_gate_upper,
        added_gate_upper=added_gate_upper,
    )


def _control_value(control: torch.Tensor) -> torch.Tensor:
    return control.detach()


def collect_gate_state(model: PositiveQuadraticBernsteinModel) -> dict:
    """Collect directly trained Bernstein control parameters."""
    added = {}
    for name in model.ADDED_GATE_NAMES:
        added[name] = {
            "beta_0": 0.0,
            "beta_1": _control_value(getattr(model, name)),
            "beta_2": 1.0,
        }

    def collect_hydro(params, channels):
        state = {}
        for channel, (beta_0_name, beta_1_name) in channels.items():
            state[channel] = {
                "beta_0": _control_value(params[beta_0_name]),
                "beta_1": _control_value(params[beta_1_name]),
                "beta_2": 1.0,
            }
        return state

    return {
        "form": "positive_quadratic_bernstein_eta",
        "variable": "coefficient-specific eta (eta_S, eta_S*eta_b, or eta_S*eta_b^2)",
        "positive_epsilon": model.POSITIVE_EPS,
        "hydro_gate_upper": model.hydro_gate_upper,
        "added_gate_upper": model.added_gate_upper,
        "hydrodynamic_force": collect_hydro(model.k_g2_X5, model.HYDRO_X5_CHANNELS),
        "hydrodynamic_moment": collect_hydro(model.k_g2_X6, model.HYDRO_X6_CHANNELS),
        "added_mass_inertia": added,
        "control_parameters": {
            "added_mass_inertia": {
                name: getattr(model, name).detach() for name in model.ADDED_GATE_NAMES
            },
            "k_g2_X5": {name: value.detach() for name, value in model.k_g2_X5.items()},
            "k_g2_X6": {name: value.detach() for name, value in model.k_g2_X6.items()},
        },
    }


def _fmt(value: torch.Tensor) -> str:
    return f"{float(value.detach().cpu().reshape(())):.6f}"


def print_gate_snapshot(model: PositiveQuadraticBernsteinModel) -> None:
    """Log the directly optimized Bernstein control values."""
    print("[eval-pre] added-mass/inertia beta_1 controls:", flush=True)
    for name in model.ADDED_GATE_NAMES:
        value = _control_value(getattr(model, name))
        print(f"[eval-pre]   {name}: [0, {_fmt(value)}, 1]", flush=True)

    for title, params, channels in (
        ("hydrodynamic force", model.k_g2_X5, model.HYDRO_X5_CHANNELS),
        ("hydrodynamic moment", model.k_g2_X6, model.HYDRO_X6_CHANNELS),
    ):
        print(f"[eval-pre] {title} [beta_0, beta_1, 1] controls:", flush=True)
        for channel, (beta_0_name, beta_1_name) in channels.items():
            beta_0 = _control_value(params[beta_0_name])
            beta_1 = _control_value(params[beta_1_name])
            print(f"[eval-pre]   {channel}: [{_fmt(beta_0)}, {_fmt(beta_1)}, 1]", flush=True)


def maybe_run_init_check(args, model, data) -> float:
    """Optionally evaluate the nominal Bernstein initialization."""
    if not args.init_check:
        return float("nan")
    init_test_loss = rollout_loss_eval(model, data, args)["test_traj_loss"]
    print(f"[init-check] initial test loss = {init_test_loss:.6e}")
    return float(init_test_loss)


def main() -> None:
    args = build_parser().parse_args()
    base_dir = os.path.dirname(__file__)
    run_dir, log_path = start_stage("stage_e_bernstein", args.out_dir, sys.argv, base_dir=base_dir)
    best_checkpoint_path = os.path.join(run_dir, "best_checkpoint.pt")
    hydro_gate_upper = (
        float(args.hydro_gate_upper)
        if args.hydro_gate_upper is not None
        else float(args.gate_upper)
    )
    added_gate_upper = (
        float(args.added_gate_upper)
        if args.added_gate_upper is not None
        else float(args.gate_upper)
    )

    stage_b = load_stage_json(args.input_b_json)
    stage_d = load_stage_json(args.input_d_json)
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
    apply_wing_gd_state(model_params, stage_d)
    model = build_bernstein_model(
        device=args.device,
        params=model_params,
        train_added=(not args.disable_added_mass_gates),
        train_hydro=True,
        hydro_gate_upper=hydro_gate_upper,
        added_gate_upper=added_gate_upper,
    )

    init_test_loss = maybe_run_init_check(args, model, data)
    named_parameter_lrs = stage_e_named_parameter_lrs(args)
    stats = run_gd_training(
        model=model,
        data=data,
        args=args,
        eval_param_callback=print_gate_snapshot,
        best_checkpoint_path=best_checkpoint_path,
        restore_best_state=True,
        parameter_projection_callback=lambda active_model: active_model.project_controls_(),
        named_parameter_lrs=named_parameter_lrs,
    )

    test_rollout_plot_paths = []
    if args.plot_test_rollouts:
        test_rollout_plot_paths = plot_test_rollout_windows(
            model,
            data.test_traj,
            data.dt,
            args.plot_test_rollout_batch_size or args.batch_size_max,
            os.path.join(run_dir, "test_rollout_plots"),
            prefix="stage_e_bernstein_test_rollout",
            batch_skip=args.batch_skip,
            num_windows=args.plot_test_rollout_windows,
        )

    if args.plot:
        plot_convergence(
            stats.get("train_loss_hist", []),
            stats.get("test_loss_hist", []),
            out_dir=run_dir,
            fname="convergence.png",
        )

    window_batch_size = args.batch_size_max
    if args.eval_windows:
        init_model = build_bernstein_model(
            device=args.device,
            params=model_params,
            train_added=False,
            train_hydro=False,
            hydro_gate_upper=hydro_gate_upper,
            added_gate_upper=added_gate_upper,
        )
        tuned_df = eval_windows_rmse(
            model,
            data.test_traj,
            dt=data.dt,
            batch_size=window_batch_size,
            batch_skip=args.batch_skip,
            norm_scale_1_13=data.norm_scale,
        )
        save_window_eval(tuned_df, run_dir, prefix="tuned")
        plot_window_rmse_boxplot(tuned_df, run_dir, fname="tuned_rmse_boxplot.png")
        init_df = eval_windows_rmse(
            init_model,
            data.test_traj,
            dt=data.dt,
            batch_size=window_batch_size,
            batch_skip=args.batch_skip,
            norm_scale_1_13=data.norm_scale,
        )
        save_window_eval(init_df, run_dir, prefix="init")
        plot_window_rmse_boxplot(init_df, run_dir, fname="init_rmse_boxplot.png")

    meta = build_gd_meta(args, data)
    meta.update(
        {
            "stage": "e_bernstein",
            "gate_form": "positive_quadratic_bernstein_eta",
            "wing_model": "finite_angle_single_wing_rate_split_v1",
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
            "gate_inputs": {
                "force": "eta_S",
                "pitch_static_and_q_rate": "eta_S",
                "roll_yaw_static": "eta_S*eta_b",
                "roll_yaw_rate": "eta_S*eta_b^2",
                "added_mass": "eta_S",
                "added_inertia_roll_yaw": "eta_S*eta_b^2",
                "added_inertia_pitch": "eta_S",
            },
            "gate_equations": {
                "hydrodynamic": "beta_0*(1-eta)^2 + 2*beta_1*eta*(1-eta) + eta^2",
                "added_mass_inertia": "2*beta_1*eta*(1-eta) + eta^2",
            },
            "gate_constraints": {
                "hydrodynamic": "eps <= beta_0,beta_1 <= hydro_gate_upper, beta_2 = 1",
                "added_mass_inertia": "beta_0 = 0, eps <= beta_1 <= added_gate_upper, beta_2 = 1",
                "note": "Added-mass/inertia gate equals zero at required eta=0 endpoint.",
                "optimization": "direct control-point training with post-step box projection",
            },
            "gate_upper": float(args.gate_upper),
            "hydro_gate_upper": hydro_gate_upper,
            "added_gate_upper": added_gate_upper,
            "gate_lr_overrides": gate_lr_meta(args),
            "log_path": log_path,
            "input_b_json": os.path.abspath(args.input_b_json),
            "input_d_json": os.path.abspath(args.input_d_json),
            "disable_added_mass_gates": bool(args.disable_added_mass_gates),
            "init_check": bool(args.init_check),
            "plot": bool(args.plot),
            "eval_windows": bool(args.eval_windows),
            "window_batch_size": window_batch_size,
            "best_checkpoint_path": best_checkpoint_path,
            "restore_best_state": True,
            "test_rollout_plot_paths": test_rollout_plot_paths,
        }
    )
    if args.init_check and init_test_loss == init_test_loss:
        meta["init_test_loss"] = init_test_loss

    result = {
        "meta": meta,
        "wing_bernstein_gates": collect_gate_state(model),
        "stats": stats,
    }
    out_path = os.path.join(run_dir, "stage_e_result.json")
    save_json(out_path, result)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
