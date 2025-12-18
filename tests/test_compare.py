"""Tests for the compare module."""

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from h5addiff.compare import compare_h5ad


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_adata():
    """Create a simple AnnData object for testing."""
    n_obs = 100
    n_vars = 50

    X = np.random.randn(n_obs, n_vars)
    obs = pd.DataFrame(
        {
            "cell_type": np.random.choice(["A", "B", "C"], n_obs),
            "n_counts": np.random.randint(1000, 10000, n_obs),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {
            "gene_name": [f"gene_{i}" for i in range(n_vars)],
            "highly_variable": np.random.choice([True, False], n_vars),
        },
        index=[f"gene_{i}" for i in range(n_vars)],
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_pca"] = np.random.randn(n_obs, 10)
    adata.uns["analysis_params"] = {"n_neighbors": 15}

    return adata


def test_identical_files(temp_dir, simple_adata):
    """Test that identical files are detected as identical."""
    file1 = temp_dir / "file1.h5ad"
    file2 = temp_dir / "file2.h5ad"

    simple_adata.write_h5ad(file1)
    simple_adata.write_h5ad(file2)

    diff = compare_h5ad(file1, file2)

    assert diff.is_identical
    assert diff.n_obs_diff == 0
    assert diff.n_vars_diff == 0
    assert diff.obs_names_equal
    assert diff.var_names_equal


def test_different_x_matrix(temp_dir, simple_adata):
    """Test detection of differences in X matrix."""
    file1 = temp_dir / "file1.h5ad"
    file2 = temp_dir / "file2.h5ad"

    simple_adata.write_h5ad(file1)

    # Modify X and save
    simple_adata.X[0, 0] = 999.0
    simple_adata.write_h5ad(file2)

    diff = compare_h5ad(file1, file2)

    assert not diff.is_identical
    assert diff.x_diff is not None
    assert not diff.x_diff.values_equal


def test_different_dimensions(temp_dir, simple_adata):
    """Test detection of different dimensions."""
    file1 = temp_dir / "file1.h5ad"
    file2 = temp_dir / "file2.h5ad"

    simple_adata.write_h5ad(file1)

    # Create smaller adata
    small_adata = simple_adata[:50, :25].copy()
    small_adata.write_h5ad(file2)

    diff = compare_h5ad(file1, file2)

    assert not diff.is_identical
    assert diff.n_obs_diff == 50
    assert diff.n_vars_diff == 25


def test_different_obs_columns(temp_dir, simple_adata):
    """Test detection of different obs columns."""
    file1 = temp_dir / "file1.h5ad"
    file2 = temp_dir / "file2.h5ad"

    simple_adata.write_h5ad(file1)

    # Add a new column
    simple_adata.obs["new_column"] = "value"
    simple_adata.write_h5ad(file2)

    diff = compare_h5ad(file1, file2)

    assert not diff.is_identical
    assert diff.obs_diff is not None
    assert not diff.obs_diff.values_equal
    assert "new_column" in diff.obs_diff.details.get("columns_only_in_second", [])


def test_different_obsm_keys(temp_dir, simple_adata):
    """Test detection of different obsm keys."""
    file1 = temp_dir / "file1.h5ad"
    file2 = temp_dir / "file2.h5ad"

    simple_adata.write_h5ad(file1)

    # Add new obsm
    simple_adata.obsm["X_umap"] = np.random.randn(simple_adata.n_obs, 2)
    simple_adata.write_h5ad(file2)

    diff = compare_h5ad(file1, file2)

    assert not diff.is_identical
    assert "X_umap" in diff.obsm_diff
    assert not diff.obsm_diff["X_umap"].exists_in_first
    assert diff.obsm_diff["X_umap"].exists_in_second
