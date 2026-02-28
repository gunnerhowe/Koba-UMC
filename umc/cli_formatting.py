"""Rich CLI formatting helpers for UMC commands."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

console = Console()


def print_header(title: str):
    """Print a styled section header."""
    console.print(Panel(Text(title, style="bold cyan"), border_style="dim"))


def print_encode_results(n_windows: int, shape: tuple, raw_bytes: int,
                         mnf_bytes: int, storage_mode: str, output: str):
    """Print encode results as a rich table."""
    ratio = raw_bytes / max(mnf_bytes, 1)
    table = Table(title="Encode Results", border_style="cyan", show_header=False, padding=(0, 2))
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Windows", str(n_windows))
    table.add_row("Shape", str(shape))
    table.add_row("Raw size", f"{raw_bytes:,} bytes")
    table.add_row(".mnf size", f"{mnf_bytes:,} bytes")
    table.add_row("Compression", f"[green]{ratio:.1f}x[/green]")
    table.add_row("Mode", storage_mode)
    table.add_row("Output", output)
    console.print(table)


def print_decode_results(n_windows: int, shape: tuple, dtype: str, output: str):
    """Print decode results."""
    table = Table(title="Decode Results", border_style="cyan", show_header=False, padding=(0, 2))
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Windows", str(n_windows))
    table.add_row("Shape", str(shape))
    table.add_row("Dtype", dtype)
    table.add_row("Output", output)
    console.print(table)


def print_stats(stats: dict):
    """Print compression statistics as a rich table."""
    table = Table(title="Compression Statistics", border_style="cyan", padding=(0, 2))
    table.add_column("Tier", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Ratio", justify="right", style="green")
    table.add_column("Per Window", justify="right", style="dim")

    table.add_row(
        "Raw data",
        f"{stats['raw_bytes']:>12,} bytes",
        "1.0x",
        f"{stats['raw_bytes'] / max(stats.get('n_windows', stats['raw_bytes'] // (stats.get('total_bytes', 1) or 1)), 1):.0f} B/win",
    )
    table.add_row(
        "VQ search",
        f"{stats['vq_bytes']:>12,} bytes",
        f"{stats['search_compression']:.1f}x",
        f"{stats['vq_bytes_per_window']:.1f} B/win",
    )
    table.add_row(
        "Storage",
        f"{stats['storage_bytes']:>12,} bytes",
        f"{stats['storage_compression']:.1f}x",
        f"{stats['storage_bytes_per_window']:.1f} B/win",
    )
    table.add_row(
        "Total (both)",
        f"{stats['total_bytes']:>12,} bytes",
        f"{stats['total_compression']:.1f}x",
        f"{stats['total_bytes_per_window']:.1f} B/win",
    )
    table.add_row(
        ".mnf file",
        f"{stats['mnf_file_bytes']:>12,} bytes",
        f"[bold green]{stats['mnf_compression']:.1f}x[/bold green]",
        "",
    )

    console.print()
    console.print(table)
    console.print(f"\n  [dim]Storage mode: {stats['storage_mode']}[/dim]")


def print_search_results(result, k: int, max_queries: int = 5):
    """Print search results as a rich table."""
    n_queries = result.indices.shape[0]
    for q in range(min(n_queries, max_queries)):
        title = f"Query {q}" if n_queries > 1 else "Search Results"
        table = Table(title=title, border_style="cyan", padding=(0, 1))
        table.add_column("Rank", justify="right", style="dim")
        table.add_column("Window Index", justify="right")
        table.add_column("L2 Distance", justify="right", style="green")

        for i in range(min(k, result.indices.shape[1])):
            idx = result.indices[q, i]
            dist = result.distances[q, i]
            table.add_row(f"#{i+1}", str(idx), f"{dist:.6f}")

        console.print(table)

    if n_queries > max_queries:
        console.print(f"\n  [dim]... ({n_queries - max_queries} more queries)[/dim]")


def make_progress():
    """Create a rich progress bar for batch processing."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )
