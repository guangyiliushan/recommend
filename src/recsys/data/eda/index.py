"""Index page generator — auto-discovers dataset EDA reports and builds listing pages.

Two-level structure:
    - Global: docs/analysis/index.md — lists all datasets, links to dataset-level index
    - Dataset-level: docs/analysis/dataset-eda/{dataset_id}/index.md — lists all subsets

Single-subset datasets (no subset directories) are linked directly from global index.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def update_index_page(
    analysis_dir: Path = Path("docs/analysis"),
    site_title: str = "数据集分析报告",
) -> Path:
    """Scan dataset-eda/ and regenerate global + per-dataset index pages.

    For each dataset directory under dataset-eda/:
        - If it has a report.md directly → single-subset (TAAC2026), link directly
        - If it has */report.md → multi-subset (TAAC2025), generate dataset-level index
          and link to it from global index

    Parameters
    ----------
    analysis_dir : Path
        Root analysis directory (default: docs/analysis).
    site_title : str
        Title for the index page.

    Returns
    -------
    Path
        Path to the generated global index.md.
    """
    eda_dir = analysis_dir / "dataset-eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    # ---- Discover all dataset directories ----
    dataset_dirs = sorted(d for d in eda_dir.iterdir() if d.is_dir())
    global_entries: list[str] = []

    for ds_dir in dataset_dirs:
        dataset_id = ds_dir.name

        # Check for direct report.md (single-subset, backward compat)
        direct_report = ds_dir / "report.md"
        if direct_report.exists():
            rel = direct_report.relative_to(analysis_dir)
            global_entries.append(f"- [{dataset_id}]({rel})")
            continue

        # Check for subset directories (multi-subset)
        subset_reports = sorted(ds_dir.glob("*/report.md"))
        if not subset_reports:
            continue

        # Generate dataset-level index
        _gen_dataset_index(ds_dir, dataset_id, subset_reports, analysis_dir)

        # Link to dataset-level index
        ds_index_rel = ds_dir / "index.md"
        rel = ds_index_rel.relative_to(analysis_dir)
        global_entries.append(f"- [{dataset_id}]({rel}) ({len(subset_reports)} subsets)")

    # ---- Write global index ----
    lines = [f"# {site_title}\n"]
    lines.append("> 本页面由 `recsys-dataset-eda` 自动生成。\n")

    if not global_entries:
        lines.append("_暂无分析报告。_\n")
    else:
        lines.append("## 可用报告\n")
        lines.extend(global_entries)
        lines.append("")

    index_path = analysis_dir / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Global index updated: %s (%d datasets)", index_path, len(global_entries))
    return index_path


def _gen_dataset_index(
    ds_dir: Path,
    dataset_id: str,
    subset_reports: list[Path],
    analysis_dir: Path,
) -> Path:
    """Generate dataset-level index.md linking to all subset reports."""
    lines = [f"# {dataset_id} — 子集分析报告\n"]
    lines.append(f"> 共 {len(subset_reports)} 个子集的分析报告。\n")

    for report_path in subset_reports:
        subset_name = report_path.parent.name
        rel = report_path.relative_to(analysis_dir)
        lines.append(f"- [{subset_name}]({rel})")
    lines.append("")

    index_path = ds_dir / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Dataset index updated: %s (%d subsets)", index_path, len(subset_reports))
    return index_path
