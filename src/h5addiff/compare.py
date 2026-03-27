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
    # Reordering detection
    obs_names_same_set: bool = True  # Same observations, possibly different order
    var_names_same_set: bool = True  # Same variables, possibly different order
    obs_reordered: bool = False  # True if obs are same set but different order
    var_reordered: bool = False  # True if var are same set but different order
    x_equal_when_reordered: bool | None = None  # True if X matches after reordering

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

    @property
    def is_equivalent(self) -> bool:
        """Check if files are equivalent (same data, possibly reordered).
        
        Files are equivalent if they contain the same observations and variables
        (possibly in different order) and the X matrix matches when reordered.
        """
        if self.is_identical:
            return True
        # Must have same set of obs and var names
        if not self.obs_names_same_set or not self.var_names_same_set:
            return False
        # X must match when reordered
        if self.x_equal_when_reordered is not True:
            return False
        return True

    @property
    def reorder_status(self) -> str:
        """Get a human-readable reordering status."""
        if self.is_identical:
            return "identical"
        if self.is_equivalent:
            parts = []
            if self.obs_reordered:
                parts.append("observations")
            if self.var_reordered:
                parts.append("variables")
            return f"equivalent (reordered: {', '.join(parts)})"
        return "different"


def _compare_arrays(
    arr1: Any,
    arr2: Any,
    name: str,
    include_sum: bool = False,
) -> ComponentDiff:
    """Compare two numpy arrays or sparse matrices.
    
    Parameters
    ----------
    arr1 : array-like
        First array to compare.
    arr2 : array-like
        Second array to compare.
    name : str
        Name of the component being compared.
    include_sum : bool, optional
        Whether to include sum comparison (useful for X matrix read counts).
    """
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

        # Calculate sums if requested (useful for total read counts)
        if include_sum:
            sum1 = float(np.nansum(dense1))
            sum2 = float(np.nansum(dense2))
            diff.details["sum_first"] = sum1
            diff.details["sum_second"] = sum2
            diff.details["sum_difference"] = sum2 - sum1
            diff.details["sum_percent_change"] = (
                float((sum2 - sum1) / sum1 * 100) if sum1 != 0 else float("inf") if sum2 != 0 else 0.0
            )

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
            
            # Add sum info to summary if available
            if include_sum:
                sum_diff = diff.details["sum_difference"]
                pct = diff.details["sum_percent_change"]
                diff.summary += f"; sum diff: {sum_diff:+.2f} ({pct:+.2f}%)"
    except Exception as e:
        diff.values_equal = False
        diff.summary = f"Comparison failed: {e}"

    return diff


def _check_x_equal_when_reordered(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    common_obs: set,
    common_var: set,
) -> bool:
    """Check if X matrices are equal when rows/columns are reordered to match.
    
    Parameters
    ----------
    adata1 : AnnData
        First AnnData object.
    adata2 : AnnData
        Second AnnData object.
    common_obs : set
        Set of observation names common to both.
    common_var : set
        Set of variable names common to both.
    
    Returns
    -------
    bool
        True if X matrices are equal after reordering.
    """
    try:
        # Get the order of obs/var names in adata1
        obs_order = list(adata1.obs_names)
        var_order = list(adata1.var_names)
        
        # Reorder adata2 to match adata1's order
        adata2_reordered = adata2[obs_order, var_order]
        
        # Compare the X matrices
        arr1: Any = adata1.X
        arr2: Any = adata2_reordered.X
        
        # Handle sparse matrices
        dense1 = arr1.toarray() if sparse.issparse(arr1) else np.asarray(arr1)
        dense2 = arr2.toarray() if sparse.issparse(arr2) else np.asarray(arr2)
        
        # Compare
        if np.issubdtype(dense1.dtype, np.floating):
            return bool(np.allclose(dense1, dense2, equal_nan=True))
        else:
            return bool(np.array_equal(dense1, dense2))
    except Exception:
        return False


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
            # Find differing columns with example differences
            differing_cols = []
            column_details: list[dict[str, Any]] = []
            for col in sorted(common_cols):
                if not df1[col].equals(df2[col]):
                    differing_cols.append(col)
                    # Find rows where values differ
                    # Cast to object to avoid Categorical comparison errors
                    s1 = df1[col].astype(object)
                    s2 = df2[col].astype(object)
                    mask = s1 != s2
                    # Also catch NaN mismatches (NaN != NaN is True but
                    # we want to flag rows where one is NaN and the other isn't)
                    try:
                        null1 = s1.isna()
                        null2 = s2.isna()
                        mask = mask | (null1 != null2)
                    except Exception:
                        pass
                    diff_indices = df1.index[mask].tolist()
                    n_rows_differ = len(diff_indices)
                    # Collect a few example differences
                    max_examples = 3
                    examples = []
                    for idx in diff_indices[:max_examples]:
                        examples.append({
                            "index": str(idx),
                            "file1": str(df1.loc[idx, col]),
                            "file2": str(df2.loc[idx, col]),
                        })
                    column_details.append({
                        "column": col,
                        "n_rows_differ": n_rows_differ,
                        "examples": examples,
                    })
            diff.details["differing_columns"] = differing_cols
            diff.details["column_details"] = column_details
            col_names = ", ".join(differing_cols)
            diff.summary = (
                f"{len(differing_cols)} columns differ: {col_names}"
            )
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

    # Check if obs/var names are the same set (possibly reordered)
    obs_set1 = set(adata1.obs_names)
    obs_set2 = set(adata2.obs_names)
    var_set1 = set(adata1.var_names)
    var_set2 = set(adata2.var_names)
    result.obs_names_same_set = obs_set1 == obs_set2
    result.var_names_same_set = var_set1 == var_set2
    result.obs_reordered = result.obs_names_same_set and not result.obs_names_equal
    result.var_reordered = result.var_names_same_set and not result.var_names_equal

    # Compare X matrix (include sum for total read count comparison)
    if adata1.X is not None and adata2.X is not None:
        result.x_diff = _compare_arrays(adata1.X, adata2.X, "X", include_sum=True)
        
        # If X differs but obs/var are same sets, check if reordering makes them equal
        if not result.x_diff.values_equal and (result.obs_reordered or result.var_reordered):
            result.x_equal_when_reordered = _check_x_equal_when_reordered(
                adata1, adata2, obs_set1, var_set1
            )
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
