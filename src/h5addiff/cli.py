"""Command-line interface for h5addiff."""

import argparse
import sys
from pathlib import Path

from . import __version__
from .compare import compare_h5ad
from .report import DiffReport


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="h5addiff",
        description="Compare two h5ad files and summarise their differences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  h5addiff file1.h5ad file2.h5ad
  h5addiff file1.h5ad file2.h5ad --format text
  h5addiff file1.h5ad file2.h5ad --backed  # Memory-efficient mode
        """,
    )

    parser.add_argument(
        "file1",
        type=Path,
        help="Path to the first h5ad file",
    )

    parser.add_argument(
        "file2",
        type=Path,
        help="Path to the second h5ad file",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["rich", "text"],
        default="rich",
        help="Output format (default: rich)",
    )

    parser.add_argument(
        "-b",
        "--backed",
        action="store_true",
        help="Read files in backed mode (memory efficient for large files)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only output exit code (0 if identical, 1 if different)",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def main(args: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    # Validate input files
    if not parsed.file1.exists():
        print(f"Error: File not found: {parsed.file1}", file=sys.stderr)
        return 2

    if not parsed.file2.exists():
        print(f"Error: File not found: {parsed.file2}", file=sys.stderr)
        return 2

    try:
        # Perform comparison
        diff = compare_h5ad(parsed.file1, parsed.file2, backed=parsed.backed)

        # Output results
        if not parsed.quiet:
            report = DiffReport(diff)
            if parsed.format == "rich":
                report.print_rich()
            else:
                print(report.to_text())

        # Return exit code
        return 0 if diff.is_identical else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
