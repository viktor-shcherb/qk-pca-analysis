# Data Format

## Source

Data is produced by [qk-sniffer](https://github.com/viktoroo/qk-sniffer), which hooks into LLM attention layers and captures Q/K vectors during inference.

## HuggingFace Dataset Layout

Each dataset on HuggingFace Hub contains one **config** per (layer, head, q/k) triple. Config names follow the pattern:

```
l{LL}h{HH}{q|k}
```

Examples: `l00h00q` (layer 0, head 0, queries), `l03h07k` (layer 3, head 7, keys).

## Parquet Columns

Each config's data is stored as a single parquet file with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `bucket` | int32 | Sampling bucket identifier |
| `example_id` | int32 | Which input example the vector came from |
| `position` | int32 | Token position within the sequence (0-indexed) |
| `vector` | FixedSizeList[float32] | The Q or K vector (length = head_dim) |
| `sliding_window` | int32 (nullable) | Sliding window size for local attention, `null` for global |
| `token_str` | string (optional) | Token string, only present if `capture_token_strings` was enabled |

## Loading

```python
from datasets import load_dataset

ds = load_dataset(
    "viktoroo/sniffed-qk-smollm2-360m-tr512-pre-rope",
    "l00h00q",
    split="train",
)
```

The `vector` column contains fixed-size lists of float32 values. For SmolLM2-360M, each vector has 64 dimensions (head_dim=64).
