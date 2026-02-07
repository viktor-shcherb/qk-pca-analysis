"""CLI entry points: pca-analyze and pca-visualize."""

from __future__ import annotations

import argparse
import sys


def main_analyze():
    """Entry point for ``pca-analyze``."""
    parser = argparse.ArgumentParser(
        description="Run PCA analysis on Q/K vectors from qk-sniffer."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    from qk_pca.analysis import run_analysis
    from qk_pca.config import load_config

    cfg = load_config(args.config)
    results_dir = run_analysis(cfg)
    print(f"\nDone. Results in {results_dir}")


def main_visualize():
    """Entry point for ``pca-visualize``."""
    parser = argparse.ArgumentParser(
        description="Visualize PCA projections for a single sequence."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--layer", type=int, required=True, help="Layer index.")
    parser.add_argument("--head", type=int, required=True, help="Head index.")
    parser.add_argument("--kind", type=str, required=True, choices=["q", "k"], help="Vector kind.")
    parser.add_argument("--example-id", type=int, required=True, help="Sequence example_id to highlight.")
    parser.add_argument("--output", type=str, required=True, help="Output file path (PNG or SVG).")
    parser.add_argument("--pc-x", type=int, default=0, help="PC index for x-axis (default: 0).")
    parser.add_argument("--pc-y", type=int, default=1, help="PC index for y-axis (default: 1).")
    args = parser.parse_args()

    from qk_pca.config import load_config
    from qk_pca.visualization import scatter_with_position

    cfg = load_config(args.config)
    scatter_with_position(
        cfg,
        layer=args.layer,
        head=args.head,
        kind=args.kind,
        example_id=args.example_id,
        output=args.output,
        pc_x=args.pc_x,
        pc_y=args.pc_y,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        sys.argv.pop(1)
        main_visualize()
    else:
        main_analyze()
