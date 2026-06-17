"""Run context — shared metadata across CLI, renderer, and reporter.

This module provides RunContext, a dataclass that centralizes all metadata
about an EDA run, including dataset identity, output paths, and sampling info.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from recsys.data.eda.sampler import SampleMetadata


@dataclass
class RunContext:
    """EDA run context — shared across CLI, renderer, and reporter.

    Attributes
    ----------
    dataset_id : str
        Sanitized dataset identifier (e.g. "taac2026_data_sample").
    dataset_label : str
        Human-readable dataset name (e.g. "TAAC2026 Sample (1000 rows)").
    source_type : str
        Data source type: "registry" or "file".
    source_ref : str
        Registry name or file path.
    run_tag : Optional[str]
        Optional tag for versioning (e.g. "20260617_174500").
    generated_at : str
        ISO timestamp of when the report was generated.
    sample_metadata : SampleMetadata
        Sampling audit metadata.
    output_dir : Path
        Resolved path for ECharts JSON output.
    report_path : Path
        Resolved path for the Markdown report.
    json_path : Optional[Path]
        Resolved path for structured stats JSON.
    assets_dir_rel : str
        Relative path from report directory to charts directory.
    """

    dataset_id: str
    dataset_label: str
    source_type: str  # "registry" | "file"
    source_ref: str
    run_tag: Optional[str]
    generated_at: str
    sample_metadata: SampleMetadata
    output_dir: Path
    report_path: Path
    json_path: Optional[Path]
    assets_dir_rel: str
    load_sampled: bool = False       # Whether pre-sampled at load time
    load_original_rows: int = 0      # Original row count before load-time sampling

    @classmethod
    def from_args(
        cls,
        dataset: Optional[str],
        dataset_path: Optional[str],
        dataset_id: Optional[str],
        run_tag: Optional[str],
        sample_metadata: SampleMetadata,
        output_dir: Optional[str],
        report_path: Optional[str],
        json_path: Optional[str],
        load_sampled: bool = False,
        load_original_rows: int = 0,
    ) -> RunContext:
        """Build RunContext from CLI arguments.

        Parameters
        ----------
        dataset : Optional[str]
            Registered dataset name.
        dataset_path : Optional[str]
            Path to local file.
        dataset_id : Optional[str]
            Explicit dataset ID override.
        run_tag : Optional[str]
            Optional version tag.
        sample_metadata : SampleMetadata
            Sampling metadata from the sampler.
        output_dir : Optional[str]
            User-specified output directory (or None for default).
        report_path : Optional[str]
            User-specified report path (or None for default).
        json_path : Optional[str]
            User-specified JSON path (or None for default).

        Returns
        -------
        RunContext
        """
        # Resolve dataset_id
        resolved_id = _resolve_dataset_id(dataset, dataset_path, dataset_id)

        # Resolve dataset_label
        if dataset:
            label = dataset
        elif dataset_path:
            label = Path(dataset_path).stem
        else:
            label = resolved_id

        # Append row count to label
        if sample_metadata.total_rows:
            label = f"{label} ({sample_metadata.total_rows:,} rows)"

        # Source info
        if dataset:
            source_type = "registry"
            source_ref = dataset
        else:
            source_type = "file"
            source_ref = dataset_path or "unknown"

        # Resolve paths
        base_output_dir = Path(output_dir) if output_dir else Path(f"docs/assets/figures/eda/{resolved_id}")
        base_report_path = Path(report_path) if report_path else Path(f"docs/analysis/dataset-eda/{resolved_id}/report.md")
        base_json_path = Path(json_path) if json_path else Path(f"docs/analysis/dataset-eda/{resolved_id}/summary.json")

        # Run tag handling
        if run_tag:
            # Insert run_tag into paths: <base>/<run_tag>/...
            output_dir_resolved = base_output_dir / run_tag
            report_path_resolved = base_report_path.parent / run_tag / base_report_path.name
            json_path_resolved = base_json_path.parent / run_tag / base_json_path.name
        else:
            output_dir_resolved = base_output_dir
            report_path_resolved = base_report_path
            json_path_resolved = base_json_path

        # Compute relative path from report to charts
        report_parent = report_path_resolved.parent
        assets_dir_rel = os.path.relpath(output_dir_resolved, report_parent).replace("\\", "/")

        return cls(
            dataset_id=resolved_id,
            dataset_label=label,
            source_type=source_type,
            source_ref=source_ref,
            run_tag=run_tag,
            generated_at=datetime.now(timezone.utc).isoformat(),
            sample_metadata=sample_metadata,
            output_dir=output_dir_resolved,
            report_path=report_path_resolved,
            json_path=json_path_resolved,
            assets_dir_rel=assets_dir_rel,
            load_sampled=load_sampled,
            load_original_rows=load_original_rows,
        )


def _resolve_dataset_id(
    dataset: Optional[str],
    dataset_path: Optional[str],
    dataset_id: Optional[str],
) -> str:
    """Resolve dataset ID from CLI arguments.

    Priority:
        1. --dataset-id (explicit override)
        2. --dataset (registry name)
        3. --dataset-path stem
        4. "unknown_dataset" fallback
    """
    if dataset_id:
        return _sanitize_id(dataset_id)
    if dataset:
        return _sanitize_id(dataset)
    if dataset_path:
        return _sanitize_id(Path(dataset_path).stem)
    return "unknown_dataset"


def _sanitize_id(raw: str) -> str:
    """Sanitize dataset ID: lowercase, replace non-alphanumeric with underscores."""
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', raw).lower()
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    return sanitized.strip('_')
