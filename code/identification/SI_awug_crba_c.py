
import argparse
import os
import sys

from SI_awug_crba_utils import (
    apply_body_state,
    build_wing_ls_residual,
    build_model,
    clone_base_params,
    load_stage_data,
    load_stage_json,
    loss_delta,
    nan_loss_dict,
    pretty_vector,
    rollout_loss_eval,
    save_json,
    start_stage,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage C command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Stage C residual least-squares identification for wing hydrodynamic coefficients."
            " / Wing hydrodynamic coefficients from residual least squares."
        )
    )
    parser.add_argument("--input-b-json", required=True, help="Path to stage_b_result.json.")
    parser.add_argument("--train-files", required=True)
    parser.add_argument("--test-files", required=True)
    parser.add_argument("--out-dir", default="log/si_awug_crba")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--dt-base", type=float, default=1.0 / 90.0)
    parser.add_argument("--norm-mode", choices=["none", "minmax", "zscore"], default="none")
    parser.add_argument("--norm-stats-json", default="norm_stats_minmax.json")
    parser.add_argument("--loss-weight-mode", choices=["none", "variance", "task", "variance_task"], default="none")
    parser.add_argument("--variance-weights-json", default="")
    parser.add_argument("--task-weights-spec", default="")
    parser.add_argument("--batch-size-max", type=int, default=360)
    parser.add_argument("--batch-skip", type=int, default=360)
    parser.add_argument(
        "--skip-input-b-eval",
        action="store_true",
        help="Skip evaluating the input Stage-B model on Stage-C train/test data.",
    )
    parser.add_argument("--ls-method", choices=["batch", "rls"], default="batch")
    parser.add_argument("--rls-lambda", type=float, default=1.0)
    parser.add_argument("--rls-delta", type=float, default=1e6)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--unweighted", action="store_true")
    parser.add_argument("--ls-col-normalize", action="store_true")
    parser.add_argument(
        "--joint-wing-ls",
        action="store_true",
        help=(
            "Use the legacy joint 6-D wing-wrench LS. In batch mode, the default "
            "Stage C behavior first "
            "identifies Theta_5 from force rows, then identifies Theta_6 from "
            "moment rows after subtracting the force-induced moment."
        ),
    )
    parser.add_argument(
        "--ridge-lambda",
        type=float,
        default=1e-3,
        help="L2 regularization strength for batch LS (0 disables ridge).",
    )
    parser.add_argument(
        "--min-linear-speed",
        type=float,
        default=0.01,
        help="Skip LS samples with |v_b| below this threshold when |w_b| is also small.",
    )
    parser.add_argument(
        "--min-angular-speed",
        type=float,
        default=0.02,
        help="Skip LS samples with |w_b| below this threshold when |v_b| is also small.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_dir = os.path.dirname(__file__)
    run_dir, log_path = start_stage("stage_c", args.out_dir, sys.argv, base_dir=base_dir)

    stage_b = load_stage_json(args.input_b_json)
    data = load_stage_data(
        train_files_arg=args.train_files,
        test_files_arg=args.test_files,
        device=args.device,
        sample_step=args.sample_step,
        dt_base=args.dt_base,
        norm_mode=args.norm_mode,
        norm_stats_json=args.norm_stats_json,
        loss_weight_mode=args.loss_weight_mode,
        variance_weights_json=args.variance_weights_json or None,
        task_weights_spec=args.task_weights_spec or None,
        base_dir=base_dir,
        include_test=True,
    )
    print(f"[data] train_files={len(data.train_files)} test_files={len(data.test_files)} dt={data.dt:.12g}")
    print(
        f"[data] train_rows={sum(int(t.shape[0]) for t in data.train_traj)} "
        f"test_rows={sum(int(t.shape[0]) for t in data.test_traj)}"
    )

    model_params = apply_body_state(clone_base_params(), stage_b)
    model = build_model(device=args.device, params=model_params)

    if args.skip_input_b_eval:
        print("[input_b] skipping input Stage-B loss eval (--skip-input-b-eval).", flush=True)
        input_b_loss = nan_loss_dict()
    else:
        print("[input_b] evaluating input Stage-B loss on Stage-C data...", flush=True)
        input_b_loss = rollout_loss_eval(model, data, args)
        print(
            f"[input_b] test_traj_loss={input_b_loss['test_traj_loss']:.6e}"
        )

    theta_5, theta_6 = build_wing_ls_residual(
        model,
        data.train_traj,
        verbose=args.verbose,
        ls_method=args.ls_method,
        rls_lambda=args.rls_lambda,
        rls_delta=args.rls_delta,
        progress_every=args.progress_every,
        weighted=(not args.unweighted),
        col_normalize=args.ls_col_normalize,
        ridge_lambda=args.ridge_lambda,
        min_linear_speed=args.min_linear_speed,
        min_angular_speed=args.min_angular_speed,
        split_force_moment=(not args.joint_wing_ls),
    )

    print("[stage_c] Theta_5:", pretty_vector(theta_5))
    print("[stage_c] Theta_6:", pretty_vector(theta_6))

    stage_c_params = apply_body_state(clone_base_params(), stage_b)
    stage_c_params["Theta_5"] = theta_5.detach().cpu().numpy()
    stage_c_params["Theta_6"] = theta_6.detach().cpu().numpy()
    stage_c_model = build_model(device=args.device, params=stage_c_params)
    print("[stage_c] evaluating Stage-C loss...", flush=True)
    stage_c_loss = rollout_loss_eval(stage_c_model, data, args)
    print(
        f"[stage_c] test_traj_loss={stage_c_loss['test_traj_loss']:.6e}"
    )

    result = {
        "meta": {
            "stage": "c",
            "input_b_json": os.path.abspath(args.input_b_json),
            "log_path": log_path,
            "train_files": data.train_files,
            "test_files": data.test_files,
            "sample_step": args.sample_step,
            "dt": data.dt,
            "dt_base": args.dt_base,
            "batch_size_max": args.batch_size_max,
            "batch_skip": args.batch_skip,
            "norm_mode": args.norm_mode,
            "norm_stats_json": args.norm_stats_json,
            "loss_weight_mode": args.loss_weight_mode,
            "variance_weights_json": args.variance_weights_json,
            "task_weights_spec": args.task_weights_spec,
            "skip_input_b_eval": bool(args.skip_input_b_eval),
            "ls_method": args.ls_method,
            "rls_lambda": args.rls_lambda,
            "rls_delta": args.rls_delta,
            "progress_every": args.progress_every,
            "weighted": (not args.unweighted),
            "col_normalize": args.ls_col_normalize,
            "split_force_moment_lsq": (not args.joint_wing_ls and args.ls_method == "batch"),
            "ridge_lambda": args.ridge_lambda,
            "min_linear_speed": args.min_linear_speed,
            "min_angular_speed": args.min_angular_speed,
            "verbose": args.verbose,
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
        },
        "wing_hydro_params": {
            "Theta_5": theta_5,
            "Theta_6": theta_6,
            # Keep raw aliases for consistency with stage-d output schema.
            "Theta_5_raw": theta_5,
            "Theta_6_raw": theta_6,
        },
        "stage_c_ls": {
            "wing_hydro_params": {
                "Theta_5": theta_5,
                "Theta_6": theta_6,
            },
            "loss": stage_c_loss,
            "loss_delta_vs_input_b": loss_delta(stage_c_loss, input_b_loss),
        },
        "input_stage_b_on_stage_c_data": {
            "skipped": bool(args.skip_input_b_eval),
            "loss": input_b_loss,
        },
    }

    out_path = os.path.join(run_dir, "stage_c_result.json")
    save_json(out_path, result)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
