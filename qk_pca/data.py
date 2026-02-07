"""HuggingFace dataset loading and head discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

# Pattern for config names produced by qk-sniffer: l{LL}h{HH}{q|k}
_CONFIG_RE = re.compile(r"^l(\d+)h(\d+)(q|k)$")


@dataclass(frozen=True)
class HeadTarget:
    """Identifies a single (layer, head, kind) triple to analyze."""

    layer: int
    head: int
    kind: str  # "q" or "k"

    @property
    def config_name(self) -> str:
        return f"l{self.layer:02d}h{self.head:02d}{self.kind}"

    def __str__(self) -> str:
        return self.config_name


def discover_available_heads(
    dataset_name: str,
    *,
    kinds: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
) -> Tuple[Optional[str], List[HeadTarget]]:
    """Discover all (layer, head, kind) configs available in a HF dataset repo.

    Uses ``huggingface_hub.list_repo_tree`` to enumerate parquet directories,
    then filters by the requested layers / heads / kinds.

    Returns
    -------
    prefix : str or None
        The directory prefix under which head configs live (e.g.
        ``"HuggingFaceTB_SmolLM2_360M"``).  ``None`` if configs sit at the
        repo root.
    targets : list[HeadTarget]
        Matched heads, sorted by (layer, head, kind).
    """
    targets: list[HeadTarget] = []
    prefix: str | None = None

    for path in list_repo_files(dataset_name, repo_type="dataset"):
        parts = path.split("/")
        for i, part in enumerate(parts):
            m = _CONFIG_RE.match(part)
            if m is None:
                continue
            # Record prefix from path components before the config dir
            if prefix is None and i > 0:
                prefix = "/".join(parts[:i])
            layer, head, kind = int(m.group(1)), int(m.group(2)), m.group(3)
            target = HeadTarget(layer=layer, head=head, kind=kind)
            if target not in targets:
                targets.append(target)
            break

    # Apply filters
    if kinds is not None:
        targets = [t for t in targets if t.kind in kinds]
    if layers is not None:
        targets = [t for t in targets if t.layer in layers]
    if heads is not None:
        targets = [t for t in targets if t.head in heads]

    targets.sort(key=lambda t: (t.layer, t.head, t.kind))
    return prefix, targets


def resolve_split(dataset_name: str, split: Optional[str]) -> Optional[str]:
    """Return the effective split/prefix for downloading data.

    If *split* is already set, return it as-is.  Otherwise, run a lightweight
    discovery to find the directory prefix (e.g. ``"HuggingFaceTB_SmolLM2_360M"``).
    """
    if split is not None:
        return split
    prefix, _ = discover_available_heads(dataset_name)
    return prefix


def _download_parquet(
    dataset_name: str,
    target: HeadTarget,
    split: Optional[str] = None,
) -> str:
    """Download a single head's parquet file and return its local path."""
    if split:
        filename = f"{split}/{target.config_name}/data.parquet"
    else:
        filename = f"{target.config_name}/data.parquet"
    return hf_hub_download(
        repo_id=dataset_name,
        filename=filename,
        repo_type="dataset",
    )


def load_head_data(
    dataset_name: str,
    target: HeadTarget,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load vectors and positions for a single head from HuggingFace.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset repo id.
    target : HeadTarget
        Which head to load.
    split : str or None
        Directory prefix inside the repo (e.g. ``"HuggingFaceTB_SmolLM2_360M"``).
        ``None`` means configs sit at the repo root.
    max_samples : int or None
        Limit the number of rows loaded.

    Returns
    -------
    vectors : ndarray of shape (N, head_dim), float32
    positions : ndarray of shape (N,), int32
    """
    local_path = _download_parquet(dataset_name, target, split)
    df = pd.read_parquet(local_path)

    if max_samples is not None and len(df) > max_samples:
        df = df.head(max_samples)

    vectors = np.stack(df["vector"].values).astype(np.float32)
    positions = df["position"].values.astype(np.int32)
    return vectors, positions


def load_head_dataframe(
    dataset_name: str,
    target: HeadTarget,
    split: Optional[str] = None,
) -> pd.DataFrame:
    """Load the full parquet data for a single head as a DataFrame.

    Useful when you need columns beyond vectors and positions
    (e.g. ``example_id`` for visualization).
    """
    local_path = _download_parquet(dataset_name, target, split)
    return pd.read_parquet(local_path)
