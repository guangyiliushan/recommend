"""Index page generator — auto-discovers dataset EDA reports and builds a listing page.

After each CLI run, update_index_page() scans docs/analysis/dataset-eda/*/report.md
and regenerates docs/analysis/index.md with links to all available reports.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def update_index_page(
    analysis_dir: Path = Path("docs/analysis"),
    site_title: str = "数据集分析报告",
) -> Path:
    """Scan dataset-eda/ and regenerate index.md.

    Parameters
    ----------
    analysis_dir : Path
        Root analysis directory (default: docs/analysis).
    site_title : str
        Title for the index page.

    Returns
    -------
    Path
        Path to the generated index.md.
    """
    eda_dir = analysis_dir / "dataset-eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    reports = sorted(eda_dir.glob("*/report.md"))

    lines = [f"# {site_title}\n"]
    lines.append("> 本页面由 `recsys-dataset-eda` 自动生成。\n")

    if not reports:
        lines.append("_暂无分析报告。_\n")
    else:
        lines.append("## 可用报告\n")
        for report_path in reports:
            dataset_id = report_path.parent.name
            rel_path = report_path.relative_to(analysis_dir)
            lines.append(f"- [{dataset_id}]({rel_path})")
        lines.append("")

    index_path = analysis_dir / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Index page updated: %s (%d reports)", index_path, len(reports))
    return index_path
