"""Report generation for h5ad diffs."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .compare import H5adDiff, ComponentDiff


class DiffReport:
    """Generate human-readable reports from H5adDiff results."""

    def __init__(self, diff: H5adDiff):
        self.diff = diff
        self.console = Console()

    def _status_icon(self, equal: bool | None) -> str:
        """Return a status icon based on equality."""
        if equal is None:
            return "❓"
        return "✓" if equal else "✗"

    @staticmethod
    def _append_column_details_text(
        comp: ComponentDiff | None, lines: list[str]
    ) -> None:
        """Append per-column diff details to the text report."""
        if comp is None or comp.values_equal:
            return
        column_details = comp.details.get("column_details")
        if not column_details:
            return
        for cd in column_details:
            col = cd["column"]
            n = cd["n_rows_differ"]
            lines.append(f"    Column '{col}': {n} rows differ")
            for ex in cd["examples"]:
                lines.append(
                    f"      {ex['index']}: {ex['file1']!s} → {ex['file2']!s}"
                )
            remaining = n - len(cd["examples"])
            if remaining > 0:
                lines.append(f"      ... and {remaining} more")

    def _format_component_row(self, comp: ComponentDiff) -> tuple:
        """Format a component diff as a table row."""
        status = self._status_icon(comp.values_equal)

        if not comp.exists_in_first:
            existence = "Only in second"
        elif not comp.exists_in_second:
            existence = "Only in first"
        else:
            existence = "Both"

        shape = ""
        if comp.shape_first and comp.shape_second:
            if comp.shape_first == comp.shape_second:
                shape = str(comp.shape_first)
            else:
                shape = f"{comp.shape_first} → {comp.shape_second}"
        elif comp.shape_first:
            shape = f"{comp.shape_first} → ∅"
        elif comp.shape_second:
            shape = f"∅ → {comp.shape_second}"

        return (status, comp.name, existence, shape, comp.summary)

    def to_text(self) -> str:
        """Generate a plain text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("H5AD File Comparison Report")
        lines.append("=" * 60)
        lines.append(f"\nFile 1: {self.diff.file1}")
        lines.append(f"File 2: {self.diff.file2}")
        lines.append("")

        # Overall status
        if self.diff.is_identical:
            lines.append("Status: ✓ Files are IDENTICAL")
        elif self.diff.is_equivalent:
            lines.append("Status: ≈ Files are EQUIVALENT (same data, different order)")
            if self.diff.obs_reordered:
                lines.append("  - Observations are reordered")
            if self.diff.var_reordered:
                lines.append("  - Variables are reordered")
        else:
            lines.append("Status: ✗ Files are DIFFERENT")

        lines.append("")
        lines.append("-" * 60)
        lines.append("DIMENSIONS")
        lines.append("-" * 60)
        lines.append(f"Observations (n_obs) difference: {self.diff.n_obs_diff}")
        lines.append(f"Variables (n_vars) difference: {self.diff.n_vars_diff}")
        lines.append(f"Observation names equal: {self.diff.obs_names_equal}")
        lines.append(f"Variable names equal: {self.diff.var_names_equal}")
        if self.diff.obs_reordered:
            lines.append("Observation names same set: ✓ (reordered)")
        if self.diff.var_reordered:
            lines.append("Variable names same set: ✓ (reordered)")

        # X matrix
        if self.diff.x_diff:
            lines.append("")
            lines.append("-" * 60)
            lines.append("X MATRIX")
            lines.append("-" * 60)
            lines.append(f"Equal: {self._status_icon(self.diff.x_diff.values_equal)}")
            if self.diff.x_diff.summary:
                lines.append(f"Details: {self.diff.x_diff.summary}")
            # Show sum comparison (total read counts)
            details = self.diff.x_diff.details
            if "sum_first" in details:
                lines.append(f"Sum (file 1): {details['sum_first']:,.2f}")
                lines.append(f"Sum (file 2): {details['sum_second']:,.2f}")
                lines.append(f"Sum difference: {details['sum_difference']:+,.2f} ({details['sum_percent_change']:+.2f}%)")

        # Helper to format component diffs
        def format_section(title: str, diffs: dict[str, ComponentDiff] | ComponentDiff | None):
            if diffs is None:
                return
            if isinstance(diffs, ComponentDiff):
                diffs = {"": diffs}
            if not diffs:
                return

            lines.append("")
            lines.append("-" * 60)
            lines.append(title)
            lines.append("-" * 60)

            for key, comp in diffs.items():
                status = self._status_icon(comp.values_equal)
                name = comp.name if comp.name else key
                lines.append(f"  {status} {name}: {comp.summary}")

        format_section("OBS (Cell Metadata)", self.diff.obs_diff)
        self._append_column_details_text(self.diff.obs_diff, lines)
        format_section("VAR (Gene Metadata)", self.diff.var_diff)
        self._append_column_details_text(self.diff.var_diff, lines)
        format_section("LAYERS", self.diff.layers_diff)
        format_section("OBSM (Cell Embeddings)", self.diff.obsm_diff)
        format_section("VARM (Gene Embeddings)", self.diff.varm_diff)
        format_section("OBSP (Cell Graphs)", self.diff.obsp_diff)
        format_section("VARP (Gene Graphs)", self.diff.varp_diff)
        format_section("UNS (Unstructured)", self.diff.uns_diff)

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def print_rich(self) -> None:
        """Print a rich-formatted report to the console."""
        # Header
        if self.diff.is_identical:
            status_panel = Panel(
                Text("✓ Files are IDENTICAL", style="bold green"),
                title="Status",
                border_style="green",
            )
        elif self.diff.is_equivalent:
            reorder_parts = []
            if self.diff.obs_reordered:
                reorder_parts.append("observations")
            if self.diff.var_reordered:
                reorder_parts.append("variables")
            reorder_info = ", ".join(reorder_parts)
            status_panel = Panel(
                Text(f"≈ Files are EQUIVALENT\n(reordered: {reorder_info})", style="bold yellow"),
                title="Status",
                border_style="yellow",
            )
        else:
            status_panel = Panel(
                Text("✗ Files are DIFFERENT", style="bold red"),
                title="Status",
                border_style="red",
            )

        self.console.print()
        self.console.print(Panel("[bold]H5AD File Comparison[/bold]"))
        self.console.print(f"[dim]File 1:[/dim] {self.diff.file1}")
        self.console.print(f"[dim]File 2:[/dim] {self.diff.file2}")
        self.console.print()
        self.console.print(status_panel)

        # Dimensions table
        dim_table = Table(title="Dimensions", show_header=True)
        dim_table.add_column("Property")
        dim_table.add_column("Value")
        dim_table.add_column("Status")

        dim_table.add_row(
            "n_obs difference",
            str(self.diff.n_obs_diff),
            "✓" if self.diff.n_obs_diff == 0 else "✗",
        )
        dim_table.add_row(
            "n_vars difference",
            str(self.diff.n_vars_diff),
            "✓" if self.diff.n_vars_diff == 0 else "✗",
        )
        dim_table.add_row(
            "obs_names equal",
            str(self.diff.obs_names_equal),
            "✓" if self.diff.obs_names_equal else "✗",
        )
        dim_table.add_row(
            "var_names equal",
            str(self.diff.var_names_equal),
            "✓" if self.diff.var_names_equal else "✗",
        )

        self.console.print()
        self.console.print(dim_table)

        # X matrix sum comparison table (if available)
        if self.diff.x_diff and "sum_first" in self.diff.x_diff.details:
            details = self.diff.x_diff.details
            sum_table = Table(title="X Matrix (Total Counts)", show_header=True)
            sum_table.add_column("Property")
            sum_table.add_column("Value", justify="right")

            sum_table.add_row("Sum (file 1)", f"{details['sum_first']:,.2f}")
            sum_table.add_row("Sum (file 2)", f"{details['sum_second']:,.2f}")
            sum_table.add_row(
                "Difference",
                f"{details['sum_difference']:+,.2f} ({details['sum_percent_change']:+.2f}%)",
            )

            self.console.print()
            self.console.print(sum_table)

        # Components table
        components_table = Table(title="Components", show_header=True)
        components_table.add_column("Status", width=6)
        components_table.add_column("Component")
        components_table.add_column("Presence")
        components_table.add_column("Shape")
        components_table.add_column("Details")

        # Add X matrix
        if self.diff.x_diff:
            components_table.add_row(*self._format_component_row(self.diff.x_diff))

        # Add obs/var
        if self.diff.obs_diff:
            components_table.add_row(*self._format_component_row(self.diff.obs_diff))
        if self.diff.var_diff:
            components_table.add_row(*self._format_component_row(self.diff.var_diff))

        # Add all other components
        for comp in self.diff.layers_diff.values():
            components_table.add_row(*self._format_component_row(comp))
        for comp in self.diff.obsm_diff.values():
            components_table.add_row(*self._format_component_row(comp))
        for comp in self.diff.varm_diff.values():
            components_table.add_row(*self._format_component_row(comp))
        for comp in self.diff.obsp_diff.values():
            components_table.add_row(*self._format_component_row(comp))
        for comp in self.diff.varp_diff.values():
            components_table.add_row(*self._format_component_row(comp))
        for comp in self.diff.uns_diff.values():
            components_table.add_row(*self._format_component_row(comp))

        self.console.print()
        self.console.print(components_table)

        # Per-column detail tables for obs and var
        self._print_column_details_rich(self.diff.obs_diff, "obs")
        self._print_column_details_rich(self.diff.var_diff, "var")

        self.console.print()

    def _print_column_details_rich(
        self, comp: ComponentDiff | None, label: str
    ) -> None:
        """Print a rich table with per-column diff examples."""
        if comp is None or comp.values_equal:
            return
        column_details = comp.details.get("column_details")
        if not column_details:
            return

        table = Table(
            title=f"{label} Column Differences",
            show_header=True,
        )
        table.add_column("Column")
        table.add_column("Rows differ", justify="right")
        table.add_column("Example index")
        table.add_column("File 1")
        table.add_column("File 2")

        for cd in column_details:
            col = cd["column"]
            n = cd["n_rows_differ"]
            examples = cd["examples"]
            if examples:
                # First example gets the column name and count
                first = examples[0]
                table.add_row(
                    col, str(n), first["index"], first["file1"], first["file2"]
                )
                for ex in examples[1:]:
                    table.add_row(
                        "", "", ex["index"], ex["file1"], ex["file2"]
                    )
            else:
                table.add_row(col, str(n), "", "", "")

        self.console.print()
        self.console.print(table)
