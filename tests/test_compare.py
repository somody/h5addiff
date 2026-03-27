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


def test_different_obs_values(temp_dir, simple_adata):
    """Test that differing obs column values are named with examples."""
    file1 = temp_dir / "file1.h5ad"
    file2 = temp_dir / "file2.h5ad"

    simple_adata.write_h5ad(file1)

    # Modify values in an existing column
    simple_adata.obs.loc[simple_adata.obs.index[0], "n_counts"] = -1
    simple_adata.obs.loc[simple_adata.obs.index[1], "n_counts"] = -2
    simple_adata.write_h5ad(file2)

    diff = compare_h5ad(file1, file2)

    assert not diff.is_identical
    assert diff.obs_diff is not None
    assert not diff.obs_diff.values_equal
    # The differing column should be named
    assert "n_counts" in diff.obs_diff.details.get("differing_columns", [])
    # Column details with examples should be present
    column_details = diff.obs_diff.details.get("column_details", [])
    n_counts_detail = [cd for cd in column_details if cd["column"] == "n_counts"]
    assert len(n_counts_detail) == 1
    assert n_counts_detail[0]["n_rows_differ"] >= 2
    assert len(n_counts_detail[0]["examples"]) >= 2
    # Each example should have index, file1, file2
    for ex in n_counts_detail[0]["examples"]:
        assert "index" in ex
        assert "file1" in ex
        assert "file2" in ex


def test_different_obs_values_summary_names_columns(temp_dir, simple_adata):
    """Test that the summary string names the differing columns."""
    file1 = temp_dir / "file1.h5ad"
    file2 = temp_dir / "file2.h5ad"

    simple_adata.write_h5ad(file1)

    # Replace cell_type with a plain (non-categorical) series to allow new values
    simple_adata.obs["cell_type"] = simple_adata.obs["cell_type"].astype(str)
    simple_adata.obs.loc[simple_adata.obs.index[0], "cell_type"] = "Z"
    simple_adata.obs.loc[simple_adata.obs.index[0], "n_counts"] = -1
    simple_adata.write_h5ad(file2)

    diff = compare_h5ad(file1, file2)

    assert diff.obs_diff is not None
    assert "cell_type" in diff.obs_diff.summary
    assert "n_counts" in diff.obs_diff.summary
    assert "2 columns differ" in diff.obs_diff.summary


def test_categorical_column_comparison(temp_dir):
    """Test that categorical columns with different categories are compared correctly."""
    n_obs = 10
    n_vars = 5
    X = np.ones((n_obs, n_vars))

    obs1 = pd.DataFrame(
        {"cat_col": pd.Categorical(["A", "B"] * 5)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    obs2 = pd.DataFrame(
        {"cat_col": pd.Categorical(["A", "C"] * 5)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

    adata1 = ad.AnnData(X=X, obs=obs1, var=var)
    adata2 = ad.AnnData(X=X, obs=obs2, var=var)

    file1 = temp_dir / "file1.h5ad"
    file2 = temp_dir / "file2.h5ad"
    adata1.write_h5ad(file1)
    adata2.write_h5ad(file2)

    diff = compare_h5ad(file1, file2)

    # Should not error out with "Categoricals can only be compared..."
    assert diff.obs_diff is not None
    assert not diff.obs_diff.values_equal
    assert "cat_col" in diff.obs_diff.details.get("differing_columns", [])
    assert "Comparison failed" not in diff.obs_diff.summary


def test_x_matrix_sum_comparison(temp_dir, simple_adata):
    """Test that X matrix comparison includes sum details."""
    file1 = temp_dir / "file1.h5ad"
    file2 = temp_dir / "file2.h5ad"

    simple_adata.write_h5ad(file1)

    simple_adata.X[0, 0] += 100.0
    simple_adata.write_h5ad(file2)

    diff = compare_h5ad(file1, file2)

    assert diff.x_diff is not None
    assert "sum_first" in diff.x_diff.details
    assert "sum_second" in diff.x_diff.details
    assert "sum_difference" in diff.x_diff.details
    assert "sum_percent_change" in diff.x_diff.details
    assert diff.x_diff.details["sum_difference"] == pytest.approx(100.0)


def test_reordered_observations_detected_as_equivalent(temp_dir):
    """Test that files with reordered observations are detected as equivalent."""
    n_obs = 20
    n_vars = 5
    X = np.arange(n_obs * n_vars, dtype=float).reshape(n_obs, n_vars)

    obs = pd.DataFrame(
        {"label": [f"type_{i % 3}" for i in range(n_obs)]},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

    adata1 = ad.AnnData(X=X, obs=obs, var=var)

    # Reverse the observation order
    reversed_idx = list(reversed(range(n_obs)))
    adata2 = adata1[reversed_idx].copy()

    file1 = temp_dir / "file1.h5ad"
    file2 = temp_dir / "file2.h5ad"
    adata1.write_h5ad(file1)
    adata2.write_h5ad(file2)

    diff = compare_h5ad(file1, file2)

    assert not diff.is_identical
    assert diff.obs_reordered
    assert diff.obs_names_same_set
    assert diff.is_equivalent
    assert diff.x_equal_when_reordered is True


def test_reordered_variables_detected_as_equivalent(temp_dir):
    """Test that files with reordered variables are detected as equivalent."""
    n_obs = 5
    n_vars = 10
    X = np.arange(n_obs * n_vars, dtype=float).reshape(n_obs, n_vars)

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

    adata1 = ad.AnnData(X=X, obs=obs, var=var)

    # Reverse variable order
    reversed_vars = list(reversed(range(n_vars)))
    adata2 = adata1[:, reversed_vars].copy()

    file1 = temp_dir / "file1.h5ad"
    file2 = temp_dir / "file2.h5ad"
    adata1.write_h5ad(file1)
    adata2.write_h5ad(file2)

    diff = compare_h5ad(file1, file2)

    assert not diff.is_identical
    assert diff.var_reordered
    assert diff.var_names_same_set
    assert diff.is_equivalent
    assert diff.x_equal_when_reordered is True


def test_truly_different_not_equivalent(temp_dir):
    """Test that files with genuinely different data are not marked equivalent."""
    n_obs = 10
    n_vars = 5

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

    adata1 = ad.AnnData(X=np.ones((n_obs, n_vars)), obs=obs, var=var)
    adata2 = ad.AnnData(X=np.zeros((n_obs, n_vars)), obs=obs, var=var)

    file1 = temp_dir / "file1.h5ad"
    file2 = temp_dir / "file2.h5ad"
    adata1.write_h5ad(file1)
    adata2.write_h5ad(file2)

    diff = compare_h5ad(file1, file2)

    assert not diff.is_identical
    assert not diff.is_equivalent
