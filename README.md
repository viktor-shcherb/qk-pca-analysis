# qk-pca-analysis

PCA analysis of Q/K attention vectors captured by [qk-sniffer](https://github.com/viktor-shcherb/qk-sniffer). Discovers correlations between high-variance PCA directions and token positions in LLM attention heads.

## Setup

```bash
pip install -e .
```

## Usage

### Run PCA analysis

```bash
pca-analyze --config configs/example.yaml
```

This will:
1. Discover all available (layer, head, q/k) configs in the HuggingFace dataset
2. Fit PCA on each head's vectors
3. Compute Pearson correlations between PCA projections and token positions
4. Write `correlations.csv` and `summary.json` to the results directory

### Visualize a single sequence

```bash
pca-visualize \
  --config configs/example.yaml \
  --layer 0 --head 0 --kind q \
  --example-id 0 \
  --output scatter.png
```

Produces a scatter plot with the full corpus as background and a single sequence highlighted with position-encoded color (viridis colormap).

## Output

- **`correlations.csv`**: `layer, head, kind, pca_dim, correlation, p_value, explained_variance_ratio`
- **`summary.json`**: Metadata (dataset, timestamp, device, top correlations)

## Configuration

See [configs/example.yaml](configs/example.yaml) for the full config schema. Key options:

| Section | Key | Description |
|---------|-----|-------------|
| `data.dataset_name` | HuggingFace repo ID | Dataset produced by qk-sniffer |
| `data.layers` / `data.heads` | `null` or list of ints | Filter which heads to analyze |
| `pca.n_components` | int | Number of principal components |
| `pca.backend` | `sklearn` or `torch` | PCA implementation |
| `analysis.parallel.enabled` | bool | Enable parallel processing |
| `device` | `auto`, `cuda`, `mps`, `cpu` | Compute device |

## Documentation

- [Data Format](docs/data-format.md) — Input data format from qk-sniffer
- [Analysis](docs/analysis.md) — PCA methodology and output interpretation
- [Visualization](docs/visualization.md) — Visualization usage and examples
