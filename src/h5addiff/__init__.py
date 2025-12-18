"""h5addiff - A tool to compare h5ad files."""

__version__ = "0.1.0"

from .compare import compare_h5ad, H5adDiff
from .report import DiffReport

__all__ = ["compare_h5ad", "H5adDiff", "DiffReport"]
