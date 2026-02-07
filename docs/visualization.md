# Visualization

## Scatter Plot

The `pca-visualize` command produces a scatter plot showing how a single sequence's tokens are distributed in PCA space.

### Usage

```bash
pca-visualize \
  --config configs/example.yaml \
  --layer 0 --head 0 --kind q \
  --example-id 0 \
  --output scatter.png
```

### What the Plot Shows

- **Grey background**: All vectors from the corpus, projected onto PC0 vs PC1. This shows the overall distribution of the head's representations.
- **Colored foreground**: Tokens from a single sequence (`--example-id`), with color encoding token position via the **viridis** colormap (dark purple = early positions, yellow = late positions).
- **Axes**: Labeled with the principal component index and its explained variance ratio.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pc-x` | 0 | PC index for x-axis |
| `--pc-y` | 1 | PC index for y-axis |
| `--output` | required | Output path (`.png` or `.svg`) |

### Interpreting Results

- If tokens form a clear trajectory from dark to light, position is encoded in this PCA subspace.
- If tokens cluster by position (e.g., early tokens in one region, late in another), the head may be using this subspace for positional information.
- Compare pre-RoPE and post-RoPE datasets to see the effect of rotary embeddings on positional structure.
