"""Orchestration: load -> PCA -> correlate -> save."""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from qk_pca.config import Config
from qk_pca.correlation import CorrelationResult, compute_correlations
from qk_pca.data import HeadTarget, discover_available_heads, load_head_data
from qk_pca.pca import fit_pca


def analyze_single_head(
    dataset_name: str,
    target: HeadTarget,
    split: str,
    n_components: int,
    pca_backend: str,
    device: str,
    max_samples: int | None,
) -> Tuple[HeadTarget, List[CorrelationResult], int]:
    """Analyze one head end-to-end. Top-level function so it's picklable.

    Returns (target, correlation_results, n_samples).
    """
    vectors, positions = load_head_data(
        dataset_name, target, split=split, max_samples=max_samples
    )
    pca_result = fit_pca(
        vectors,
        n_components=n_components,
        backend=pca_backend,
        device=device,
    )
    correlations = compute_correlations(
        pca_result.projections,
        positions,
        pca_result.explained_variance_ratio,
    )
    return target, correlations, len(positions)


def _resolve_max_workers(cfg: Config) -> int | None:
    """Determine the number of parallel workers."""
    par = cfg.analysis.parallel
    if not par.enabled:
        return None  # signals sequential execution
    if par.max_workers is not None:
        return par.max_workers
    return min(os.cpu_count() or 1, 4)


def run_analysis(cfg: Config) -> Path:
    """Run the full analysis pipeline and write results to disk.

    Returns the path to the results directory.
    """
    results_dir = cfg.results_path
    results_dir.mkdir(parents=True, exist_ok=True)

    # Discover heads
    print(f"Discovering heads in {cfg.data.dataset_name} ...")
    discovered_prefix, targets = discover_available_heads(
        cfg.data.dataset_name,
        kinds=cfg.data.kinds,
        layers=cfg.data.layers,
        heads=cfg.data.heads,
    )
    split = cfg.data.split or discovered_prefix
    print(f"Found {len(targets)} head configs available (prefix={split!r}).")

    # Random sampling
    if cfg.data.sample_heads is not None and len(targets) > cfg.data.sample_heads:
        rng = random.Random(cfg.data.seed)
        targets = sorted(
            rng.sample(targets, cfg.data.sample_heads),
            key=lambda t: (t.layer, t.head, t.kind),
        )
        print(f"Randomly sampled {len(targets)} heads: {[str(t) for t in targets]}")

    if not targets:
        print("No matching heads found. Check your config filters.")
        return results_dir

    # Common kwargs for each worker call
    common = dict(
        dataset_name=cfg.data.dataset_name,
        split=split,
        n_components=cfg.pca.n_components,
        pca_backend=cfg.pca.backend,
        device=cfg.device,
        max_samples=cfg.data.max_samples,
    )

    rows: list[dict] = []
    total_samples = 0
    max_workers = _resolve_max_workers(cfg)

    if max_workers is None:
        # Sequential execution
        for target in tqdm(targets, desc="Analyzing heads"):
            _target, corrs, n = analyze_single_head(target=target, **common)
            total_samples += n
            for c in corrs:
                rows.append(
                    {
                        "layer": _target.layer,
                        "head": _target.head,
                        "kind": _target.kind,
                        "pca_dim": c.pca_dim,
                        "correlation": c.correlation,
                        "p_value": c.p_value,
                        "explained_variance_ratio": c.explained_variance_ratio,
                    }
                )
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(analyze_single_head, target=target, **common): target
                for target in targets
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Analyzing heads"
            ):
                _target, corrs, n = future.result()
                total_samples += n
                for c in corrs:
                    rows.append(
                        {
                            "layer": _target.layer,
                            "head": _target.head,
                            "kind": _target.kind,
                            "pca_dim": c.pca_dim,
                            "correlation": c.correlation,
                            "p_value": c.p_value,
                            "explained_variance_ratio": c.explained_variance_ratio,
                        }
                    )

    # Write CSV
    df = pd.DataFrame(rows)
    df.sort_values(["layer", "head", "kind", "pca_dim"], inplace=True)
    csv_path = results_dir / "correlations.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(df)} rows)")

    # Find top correlations (by absolute value)
    top_n = min(20, len(df))
    top = df.reindex(df["correlation"].abs().nlargest(top_n).index)

    # High-correlation summary
    threshold = cfg.analysis.correlation_threshold
    high = df[df["correlation"].abs() >= threshold].copy()
    high_neg = high[high["correlation"] < 0]

    # Per-head: does this head have ANY high-|r| component?
    head_keys = df[["layer", "head", "kind"]].drop_duplicates()
    high_head_keys = high[["layer", "head", "kind"]].drop_duplicates()
    high_neg_head_keys = high_neg[["layer", "head", "kind"]].drop_duplicates()

    high_correlation_summary = {
        "threshold": threshold,
        "total_components": len(df),
        "high_abs_components": len(high),
        "high_neg_components": len(high_neg),
        "heads_total": len(head_keys),
        "heads_with_high_abs": len(high_head_keys),
        "heads_with_high_neg": len(high_neg_head_keys),
        "high_abs_hits": high.to_dict(orient="records"),
    }

    # Print summary
    print(
        f"\nHigh-correlation summary (|r| >= {threshold}):\n"
        f"  {len(high)}/{len(df)} components, "
        f"{len(high_head_keys)}/{len(head_keys)} heads\n"
        f"  Negative: {len(high_neg)} components, "
        f"{len(high_neg_head_keys)} heads"
    )

    # Write JSON summary
    summary = {
        "dataset": cfg.data.dataset_name,
        "split": cfg.data.split,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": cfg.device,
        "n_components": cfg.pca.n_components,
        "pca_backend": cfg.pca.backend,
        "heads_analyzed": len(targets),
        "total_samples": total_samples,
        "high_correlation_summary": high_correlation_summary,
        "top_correlations": top.to_dict(orient="records"),
    }
    json_path = results_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {json_path}")

    return results_dir
