"""Evaluate Scheme41 multi-body full model against the single-body zero-sweep model."""

from __future__ import annotations

from scale_ablation import MULTI_VS_SINGLE_VARIANTS, run_scheme41_summary


def main() -> None:
    run_scheme41_summary(
        variants=MULTI_VS_SINGLE_VARIANTS,
        output_csv="multi_vs_single_summary.csv",
        description="Evaluate Scheme41 multi-body versus single-body models.",
        default_out_dir="multi_vs_single_results",
    )


if __name__ == "__main__":
    main()
