"""Core comparison logic for h5ad files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class ComponentDiff:
    """Represents differences in a single component of an AnnData object."""

    name: str
    exists_in_first: bool = True
    exists_in_second: bool = True
    shape_first: tuple | None = None
    shape_second: tuple | None = None
    dtype_first: str | None = None
    dtype_second: str | None = None
    values_equal: bool | None = None
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class H5adDiff:
    """Complete diff result between two h5ad files."""

    file1: str
    file2: str
    n_obs_diff: int = 0
    n_vars_diff: int = 0
    obs_diff: ComponentDiff | None = None
    var_diff: ComponentDiff | None = None
    x_diff: ComponentDiff | None = None
    layers_diff: dict[str, ComponentDiff] = field(default_factory=dict)
    obsm_diff: dict[str, ComponentDiff] = field(default_factory=dict)
    varm_diff: dict[str, ComponentDiff] = field(default_factory=dict)
    obsp_diff: dict[str, ComponentDiff] = field(default_factory=dict)
    varp_diff: dict[str, ComponentDiff] = field(default_factory=dict)
    uns_diff: dict[str, ComponentDiff] = field(default_factory=dict)
    obs_names_equal: bool = True
    var_names_equal: bool = True

    @property
    def is_identical(self) -> bool:
        """Check if the two files are identical."""
        if self.n_obs_diff != 0 or self.n_vars_diff != 0:
            return False
        if not self.obs_names_equal or not self.var_names_equal:
            return False
        if self.x_diff and not self.x_diff.values_equal:
            return False
        if self.obs_diff and not self.obs_diff.values_equal:
            return False
        if self.var_diff and not self.var_diff.values_equal:
            return False
        for diff in self.layers_diff.values():
            if not diff.values_equal:
                return False
        for diff in self.obsm_diff.values():
            if not diff.values_equal:
                return False
        for diff in self.varm_diff.values():
            if not diff.values_equal:
                return False
        for diff in self.uns_diff.values():
            if not diff.values_equal:
                return False
        return True


def _compare_arrays(
    arr1: Any,
    arr2: Any,
    name: str,
) -> ComponentDiff:
    """Compare two numpy arrays or sparse matrices."""
    diff = ComponentDiff(
        name=name,
        shape_first=arr1.shape,
        shape_second=arr2.shape,
        dtype_first=str(arr1.dtype),
        dtype_second=str(arr2.dtype),
    )

    if arr1.shape != arr2.shape:
        diff.values_equal = False
        diff.summary = f"Shape mismatch: {arr1.shape} vs {arr2.shape}"
        return diff

    try:
        # Handle sparse matrices - convert to dense for comparison
        dense1 = arr1.toarray() if sparse.issparse(arr1) else np.asarray(arr1)
        dense2 = arr2.toarray() if sparse.issparse(arr2) else np.asarray(arr2)

        # Handle NaN values
        if np.issubdtype(dense1.dtype, np.floating):
            equal = np.allclose(dense1, dense2, equal_nan=True)
        else:
            equal = np.array_equal(dense1, dense2)

        diff.values_equal = equal
        if not equal:
            n_diff = int(np.sum(dense1 != dense2))
            diff.summary = f"{n_diff} values differ"
            diff.details["n_different"] = n_diff
            diff.details["percent_different"] = float(n_diff / dense1.size * 100)
    except Exception as e:
        diff.values_equal = False
        diff.summary = f"Comparison failed: {e}"

    return diff


def _compare_dataframes(
    df1: Any,
    df2: Any,
    name: str,
) -> ComponentDiff:
    """Compare two pandas DataFrames."""
    diff = ComponentDiff(
        name=name,
        shape_first=df1.shape,
        shape_second=df2.shape,
        dtype_first=str(dict(df1.dtypes)),
        dtype_second=str(dict(df2.dtypes)),
    )

    # Check columns
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    only_in_first = cols1 - cols2
    only_in_second = cols2 - cols1
    common_cols = cols1 & cols2

    diff.details["columns_only_in_first"] = list(only_in_first)
    diff.details["columns_only_in_second"] = list(only_in_second)
    diff.details["common_columns"] = list(common_cols)

    if only_in_first or only_in_second:
        diff.values_equal = False
        diff.summary = f"Column mismatch: {len(only_in_first)} only in first, {len(only_in_second)} only in second"
        return diff

    if df1.shape != df2.shape:
        diff.values_equal = False
        diff.summary = f"Shape mismatch: {df1.shape} vs {df2.shape}"
        return diff

    # Compare values
    try:
        equal = df1.equals(df2)
        diff.values_equal = equal
        if not equal:
            # Find differing columns
            differing_cols = []
            for col in common_cols:
                if not df1[col].equals(df2[col]):
                    differing_cols.append(col)
            diff.details["differing_columns"] = differing_cols
            diff.summary = f"{len(differing_cols)} columns have different values"
    except Exception as e:
        diff.values_equal = False
        diff.summary = f"Comparison failed: {e}"

    return diff


def _compare_dict_like(
    dict1: dict, dict2: dict, name: str, compare_func
) -> dict[str, ComponentDiff]:
    """Compare dictionary-like structures (layers, obsm, etc.)."""
    results = {}
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    only_in_first = keys1 - keys2
    only_in_second = keys2 - keys1
    common_keys = keys1 & keys2

    for key in only_in_first:
        results[key] = ComponentDiff(
            name=f"{name}/{key}",
            exists_in_first=True,
            exists_in_second=False,
            values_equal=False,
            summary="Only in first file",
        )

    for key in only_in_second:
        results[key] = ComponentDiff(
            name=f"{name}/{key}",
            exists_in_first=False,
            exists_in_second=True,
            values_equal=False,
            summary="Only in second file",
        )

    for key in common_keys:
        results[key] = compare_func(dict1[key], dict2[key], f"{name}/{key}")

    return results


def _compare_uns(uns1: dict, uns2: dict, prefix: str = "uns") -> dict[str, ComponentDiff]:
    """Compare unstructured annotation dictionaries."""
    results = {}
    keys1 = set(uns1.keys())
    keys2 = set(uns2.keys())
    only_in_first = keys1 - keys2
    only_in_second = keys2 - keys1
    common_keys = keys1 & keys2

    for key in only_in_first:
        results[key] = ComponentDiff(
            name=f"{prefix}/{key}",
            exists_in_first=True,
            exists_in_second=False,
            values_equal=False,
            summary="Only in first file",
        )

    for key in only_in_second:
        results[key] = ComponentDiff(
            name=f"{prefix}/{key}",
            exists_in_first=False,
            exists_in_second=True,
            values_equal=False,
            summary="Only in second file",
        )

    for key in common_keys:
        val1, val2 = uns1[key], uns2[key]

        # Handle nested dicts
        if isinstance(val1, dict) and isinstance(val2, dict):
            nested = _compare_uns(val1, val2, f"{prefix}/{key}")
            results.update(nested)
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            results[key] = _compare_arrays(val1, val2, f"{prefix}/{key}")
        elif isinstance(val1, pd.DataFrame) and isinstance(val2, pd.DataFrame):
            results[key] = _compare_dataframes(val1, val2, f"{prefix}/{key}")
        else:
            # Simple equality check
            try:
                equal = val1 == val2
                if isinstance(equal, (np.ndarray, pd.Series)):
                    equal = equal.all()
            except Exception:
                equal = False

            results[key] = ComponentDiff(
                name=f"{prefix}/{key}",
                values_equal=bool(equal),
                summary="" if equal else "Values differ",
            )

    return results


def compare_h5ad(
    file1: str | Path,
    file2: str | Path,
    backed: bool = False,
) -> H5adDiff:
    """
    Compare two h5ad files and return a detailed diff.

    Parameters
    ----------
    file1 : str or Path
        Path to the first h5ad file.
    file2 : str or Path
        Path to the second h5ad file.
    backed : bool, optional
        Whether to read files in backed mode (memory efficient for large files).

    Returns
    -------
    H5adDiff
        Detailed comparison results.
    """
    file1 = Path(file1)
    file2 = Path(file2)

    # Load the files
    mode = "r" if backed else None
    adata1 = ad.read_h5ad(file1, backed=mode)
    adata2 = ad.read_h5ad(file2, backed=mode)

    result = H5adDiff(file1=str(file1), file2=str(file2))

    # Compare dimensions
    result.n_obs_diff = adata1.n_obs - adata2.n_obs
    result.n_vars_diff = adata1.n_vars - adata2.n_vars

    # Compare obs and var names
    result.obs_names_equal = adata1.obs_names.equals(adata2.obs_names)
    result.var_names_equal = adata1.var_names.equals(adata2.var_names)

    # Compare X matrix
    if adata1.X is not None and adata2.X is not None:
        result.x_diff = _compare_arrays(adata1.X, adata2.X, "X")
    elif adata1.X is not None or adata2.X is not None:
        result.x_diff = ComponentDiff(
            name="X",
            exists_in_first=adata1.X is not None,
            exists_in_second=adata2.X is not None,
            values_equal=False,
            summary="X matrix missing in one file",
        )

    # Compare obs and var DataFrames
    result.obs_diff = _compare_dataframes(adata1.obs, adata2.obs, "obs")
    result.var_diff = _compare_dataframes(adata1.var, adata2.var, "var")

    # Compare layers
    result.layers_diff = _compare_dict_like(
        dict(adata1.layers), dict(adata2.layers), "layers", _compare_arrays
    )

    # Compare obsm and varm
    result.obsm_diff = _compare_dict_like(
        dict(adata1.obsm), dict(adata2.obsm), "obsm", _compare_arrays
    )
    result.varm_diff = _compare_dict_like(
        dict(adata1.varm), dict(adata2.varm), "varm", _compare_arrays
    )

    # Compare obsp and varp
    result.obsp_diff = _compare_dict_like(
        dict(adata1.obsp), dict(adata2.obsp), "obsp", _compare_arrays
    )
    result.varp_diff = _compare_dict_like(
        dict(adata1.varp), dict(adata2.varp), "varp", _compare_arrays
    )

    # Compare uns
    result.uns_diff = _compare_uns(dict(adata1.uns), dict(adata2.uns))

    # Close backed files
    if backed:
        adata1.file.close()
        adata2.file.close()

    return result
