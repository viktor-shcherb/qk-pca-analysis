"""Scatter plots with position-encoded opacity."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from qk_pca.config import Config
from qk_pca.data import HeadTarget, load_head_dataframe, resolve_split
from qk_pca.pca import fit_pca


def scatter_with_position(
    cfg: Config,
    layer: int,
    head: int,
    kind: str,
    example_id: int,
    output: str | Path,
    *,
    pc_x: int = 0,
    pc_y: int = 1,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 150,
) -> Path:
    """Create a scatter plot of PC projections with position-encoded color.

    The full corpus is shown as light grey background dots. A single sequence
    (identified by *example_id*) is overlaid with a viridis colormap where
    color encodes token position within the sequence.

    Parameters
    ----------
    cfg : Config
        Experiment configuration (used for dataset name, split, PCA settings).
    layer, head, kind : int, int, str
        Which attention head to visualize.
    example_id : int
        Which sequence to highlight.
    output : path
        Destination file (PNG or SVG based on extension).
    pc_x, pc_y : int
        Which principal components to plot on x / y axes.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    target = HeadTarget(layer=layer, head=head, kind=kind)
    split = resolve_split(cfg.data.dataset_name, cfg.data.split)

    # Load full corpus and fit PCA
    df = load_head_dataframe(cfg.data.dataset_name, target, split=split)
    vectors_all = np.stack(df["vector"].values).astype(np.float32)
    positions_all = df["position"].values.astype(np.int32)
    example_ids = df["example_id"].values.astype(np.int32)

    pca_result = fit_pca(
        vectors_all,
        n_components=cfg.pca.n_components,
        backend=cfg.pca.backend,
        device=cfg.device,
    )

    # Split into corpus (background) and sequence (highlighted)
    seq_mask = example_ids == example_id
    if not seq_mask.any():
        raise ValueError(
            f"example_id={example_id} not found in dataset. "
            f"Available range: {example_ids.min()}..{example_ids.max()}"
        )

    proj_all = pca_result.projections
    proj_seq = proj_all[seq_mask]
    pos_seq = positions_all[seq_mask]

    # Sort sequence by position for consistent coloring
    order = np.argsort(pos_seq)
    proj_seq = proj_seq[order]
    pos_seq = pos_seq[order]

    # Normalize positions to [0, 1] for colormap
    pos_min, pos_max = pos_seq.min(), pos_seq.max()
    if pos_max > pos_min:
        pos_norm = (pos_seq - pos_min) / (pos_max - pos_min)
    else:
        pos_norm = np.zeros_like(pos_seq, dtype=float)

    evr_x = pca_result.explained_variance_ratio[pc_x]
    evr_y = pca_result.explained_variance_ratio[pc_y]

    # Compute correlations between each PC and position (full corpus)
    pos_float = positions_all.astype(np.float64)
    r_x, p_x = pearsonr(proj_all[:, pc_x].astype(np.float64), pos_float)
    r_y, p_y = pearsonr(proj_all[:, pc_y].astype(np.float64), pos_float)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Background: all corpus points in light grey
    ax.scatter(
        proj_all[:, pc_x],
        proj_all[:, pc_y],
        s=4,
        c="#dddddd",
        alpha=0.3,
        rasterized=True,
        label="_nolegend_",
    )

    # Foreground: highlighted sequence with viridis colormap
    sc = ax.scatter(
        proj_seq[:, pc_x],
        proj_seq[:, pc_y],
        s=30,
        c=pos_norm,
        cmap="viridis",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.3,
        label="_nolegend_",
        zorder=5,
    )

    cbar = fig.colorbar(sc, ax=ax, label="position (normalized)")
    # Add actual position ticks
    if pos_max > pos_min:
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels([str(pos_min), str((pos_min + pos_max) // 2), str(pos_max)])

    ax.set_xlabel(f"PC{pc_x} ({evr_x:.1%} var, r={r_x:+.3f})")
    ax.set_ylabel(f"PC{pc_y} ({evr_y:.1%} var, r={r_y:+.3f})")
    ax.set_title(f"{target}  (n={len(proj_seq)} tokens)")

    # Correlation summary box
    corr_text = "\n".join(
        f"PC{d}: r={pearsonr(proj_all[:, d].astype(np.float64), pos_float)[0]:+.3f}"
        for d in range(pca_result.projections.shape[1])
    )
    ax.text(
        0.02, 0.98, corr_text,
        transform=ax.transAxes, fontsize=8, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output}")
    return output
