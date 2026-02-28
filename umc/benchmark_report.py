"""HTML benchmark report generator for UMC compression results."""

import html
from datetime import datetime


def generate_report(data: dict, output_path: str) -> None:
    """Generate a standalone HTML benchmark report.

    Args:
        data: Dict with keys: file, file_size, raw_size, is_raw_binary,
              results (list of UMC results), competitors (list of competitor results).
        output_path: Path to write the HTML file.
    """
    all_entries = []
    for r in data.get("results", []):
        all_entries.append({
            "name": r["name"],
            "size": r["size"],
            "ratio": r["ratio"],
            "time": r["time"],
            "lossless": r.get("lossless", True),
            "is_umc": True,
        })
    for c in data.get("competitors", []):
        all_entries.append({
            "name": c["name"],
            "size": c["size"],
            "ratio": c["ratio"],
            "time": c["time"],
            "lossless": True,
            "is_umc": False,
        })

    # Sort by ratio descending
    all_entries.sort(key=lambda x: x["ratio"], reverse=True)

    max_ratio = max((e["ratio"] for e in all_entries), default=1)
    file_name = html.escape(data.get("file", "unknown"))
    raw_size = data.get("raw_size", 0)

    # Build table rows
    rows_html = ""
    for i, e in enumerate(all_entries):
        bar_pct = (e["ratio"] / max(max_ratio, 0.01)) * 100
        speed = raw_size / max(e["time"], 1e-9) / 1e6
        cls = "umc-row" if e["is_umc"] else "std-row"
        medal = ""
        if i == 0:
            medal = " &#x1f947;"  # gold
        elif i == 1:
            medal = " &#x1f948;"  # silver
        elif i == 2:
            medal = " &#x1f949;"  # bronze

        lossless_tag = "Yes" if e["lossless"] else "No"
        rows_html += f"""        <tr class="{cls}">
            <td>{html.escape(e['name'])}{medal}</td>
            <td>{e['size']:,}</td>
            <td><strong>{e['ratio']:.2f}x</strong></td>
            <td>
                <div class="bar-container">
                    <div class="bar {'bar-umc' if e['is_umc'] else 'bar-std'}" style="width:{bar_pct:.1f}%"></div>
                </div>
            </td>
            <td>{speed:.1f} MB/s</td>
            <td>{lossless_tag}</td>
        </tr>
"""

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UMC Benchmark Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0d1117; color: #c9d1d9; padding: 2rem;
    }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    h1 {{ color: #58a6ff; margin-bottom: 0.5rem; font-size: 1.8rem; }}
    .subtitle {{ color: #8b949e; margin-bottom: 2rem; }}
    .meta {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
             padding: 1rem 1.5rem; margin-bottom: 2rem; display: flex; gap: 2rem; flex-wrap: wrap; }}
    .meta-item {{ }}
    .meta-label {{ color: #8b949e; font-size: 0.85rem; }}
    .meta-value {{ color: #c9d1d9; font-size: 1.1rem; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; background: #161b22;
             border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
    th {{ background: #21262d; color: #8b949e; padding: 0.75rem 1rem; text-align: left;
          font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    td {{ padding: 0.65rem 1rem; border-top: 1px solid #21262d; }}
    .umc-row td:first-child {{ color: #58a6ff; font-weight: 600; }}
    .std-row td:first-child {{ color: #8b949e; }}
    .bar-container {{ width: 100%; height: 20px; background: #21262d; border-radius: 4px; overflow: hidden; }}
    .bar {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
    .bar-umc {{ background: linear-gradient(90deg, #1f6feb, #58a6ff); }}
    .bar-std {{ background: linear-gradient(90deg, #30363d, #484f58); }}
    .footer {{ margin-top: 2rem; color: #484f58; font-size: 0.8rem; text-align: center; }}
    .footer a {{ color: #58a6ff; text-decoration: none; }}
</style>
</head>
<body>
<div class="container">
    <h1>UMC Benchmark Report</h1>
    <p class="subtitle">Compression comparison for your data</p>

    <div class="meta">
        <div class="meta-item">
            <div class="meta-label">File</div>
            <div class="meta-value">{file_name}</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">Raw Size</div>
            <div class="meta-value">{_fmt_size(raw_size)}</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">File Size</div>
            <div class="meta-value">{_fmt_size(data.get('file_size', 0))}</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">Generated</div>
            <div class="meta-value">{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Compressor</th>
                <th>Compressed Size</th>
                <th>Ratio</th>
                <th>Visual</th>
                <th>Speed</th>
                <th>Lossless</th>
            </tr>
        </thead>
        <tbody>
{rows_html}        </tbody>
    </table>

    <div class="footer">
        Generated by <a href="https://github.com/your-org/umc">UMC</a> (Universal Manifold Codec)
    </div>
</div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_html)


def _fmt_size(n: int) -> str:
    """Format byte count as human-readable string."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    else:
        return f"{n / (1024 * 1024 * 1024):.1f} GB"
