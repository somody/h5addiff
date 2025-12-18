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

        # X matrix
        if self.diff.x_diff:
            lines.append("")
            lines.append("-" * 60)
            lines.append("X MATRIX")
            lines.append("-" * 60)
            lines.append(f"Equal: {self._status_icon(self.diff.x_diff.values_equal)}")
            if self.diff.x_diff.summary:
                lines.append(f"Details: {self.diff.x_diff.summary}")

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
        format_section("VAR (Gene Metadata)", self.diff.var_diff)
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
        else:
            status_panel = Panel(
                Text("✗ Files are DIFFERENT", style="bold red"),
                title="Status",
                border_style="red",
            )

        self.console.print()
        self.console.print(Panel(f"[bold]H5AD File Comparison[/bold]"))
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
        self.console.print()
