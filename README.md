# h5addiff

A simple tool to determine and summarise the difference between two h5ad files.

## Overview

`h5addiff` compares two [AnnData](https://anndata.readthedocs.io/) h5ad files and generates a detailed report of their differences. It compares:

- **Dimensions**: Number of observations (cells) and variables (genes)
- **X matrix**: The main data matrix
- **obs/var**: Cell and gene metadata DataFrames
- **layers**: Additional data layers
- **obsm/varm**: Embeddings (PCA, UMAP, etc.)
- **obsp/varp**: Pairwise relationships (graphs, distances)
- **uns**: Unstructured annotations

## Installation

### From source (development)

```bash
# Clone the repository
git clone https://github.com/yourusername/h5addiff.git
cd h5addiff

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync
```

### Using uv (once published)

```bash
uv add h5addiff
```

## Usage

### Command Line

```bash
# Basic comparison
h5addiff file1.h5ad file2.h5ad

# Plain text output (instead of rich formatting)
h5addiff file1.h5ad file2.h5ad --format text

# Memory-efficient mode for large files
h5addiff file1.h5ad file2.h5ad --backed

# Quiet mode (only exit code)
h5addiff file1.h5ad file2.h5ad --quiet
```

### Exit Codes

- `0`: Files are identical
- `1`: Files are different
- `2`: Error (file not found, invalid format, etc.)

### Python API

```python
from h5addiff import compare_h5ad, DiffReport

# Compare two files
diff = compare_h5ad("file1.h5ad", "file2.h5ad")

# Check if files are identical
if diff.is_identical:
    print("Files are identical!")
else:
    print(f"Observations differ by: {diff.n_obs_diff}")
    print(f"Variables differ by: {diff.n_vars_diff}")

# Generate a report
report = DiffReport(diff)
report.print_rich()  # Rich formatted output
print(report.to_text())  # Plain text output
```

## Development

### Running Tests

```bash
# Sync dependencies (including dev)
uv sync

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=h5addiff
```

### Building

```bash
# Build the package
uv build
```

## Requirements

- Python >= 3.9
- anndata >= 0.10.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- rich >= 13.0.0

## Licence

MIT Licence - see [LICENSE](LICENSE) for details.
