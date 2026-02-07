# Analysis Methodology

## Pipeline

For each (layer, head, q/k) triple:

1. **Load** vectors and positions from HuggingFace
2. **Fit PCA** with `n_components` principal components
3. **Project** all vectors onto the principal components
4. **Correlate** each PC projection with token position using Pearson correlation

## PCA

### sklearn backend (default)

Uses `sklearn.decomposition.PCA`, which computes exact SVD via LAPACK. This is the best choice for head dimensions typical in LLMs (64-128), where the matrix is small enough for exact decomposition.

### torch backend

Uses `torch.linalg.svd` for full SVD on a specified device. Useful for GPU-accelerated PCA on very large datasets, though the sklearn backend is generally sufficient.

## Pearson Correlation

For each principal component `i`, we compute:

```
r, p = pearsonr(projections[:, i], positions)
```

- **r** (correlation): Ranges from -1 to +1. A strong positive correlation means higher positions project more strongly onto this PC direction. A strong negative correlation means the opposite.
- **p** (p-value): Statistical significance. With large sample sizes typical in these datasets, even weak correlations tend to be significant.

Linear correlation is appropriate here because PCA components are themselves linear projections, so we're measuring the linear relationship between position and the model's learned vector subspace.

## Output

### correlations.csv

One row per (layer, head, kind, pca_dim):

```csv
layer,head,kind,pca_dim,correlation,p_value,explained_variance_ratio
0,0,q,0,0.423,1.2e-50,0.312
0,0,q,1,-0.156,3.4e-10,0.187
```

### summary.json

```json
{
  "dataset": "viktoroo/sniffed-qk-smollm2-360m-tr512-pre-rope",
  "split": "train",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "device": "auto",
  "n_components": 10,
  "heads_analyzed": 64,
  "total_samples": 1048576,
  "top_correlations": [...]
}
```

## Parallelism

Each head is fully independent — it loads its own data, fits its own PCA, and computes its own correlations. We use `ProcessPoolExecutor` with `max_workers = min(cpu_count, 4)` to limit memory usage (each worker loads one head's data at a time).
