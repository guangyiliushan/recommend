"""Dataset Exploratory Data Analysis (EDA) module.

Public API:
    - EDAConfig: configuration dataclass
    - SampleMetadata: sampling metadata for audit trail
    - run_eda(): main entry point for EDA pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from recsys.data.eda.context import RunContext  # noqa: F401
from recsys.data.eda.sampler import SampleMetadata, hybrid_sample  # noqa: F401


@dataclass
class EDAConfig:
    """EDA pipeline configuration.

    Attributes
    ----------
    max_rows : int
        Maximum rows after sampling. Default 500k.
    sample_seed : int
        Random seed for reproducible sampling.
    label_col : str
        Name of the label column.
    item_col : str
        Name of the item ID column.
    user_col : str
        Name of the user ID column.
    domain_pattern : str
        Prefix pattern for domain sequence columns.
    tail_quantile : float
        Quantile threshold for tail-preserving sampling.
    output_dir : str
        Directory for ECharts JSON output.
    report_path : str
        Path for the Markdown report.
    json_path : Optional[str]
        Path for structured stats JSON output.
    json_only : bool
        If True, only output structured JSON, no charts or report.
    """

    max_rows: int = 500_000
    sample_seed: int = 42
    label_col: str = "label_type"
    item_col: str = "item_id"
    user_col: str = "user_id"
    domain_pattern: str = "domain_"
    dense_pattern: str = "user_dense_"
    tail_quantile: float = 0.95
    cold_start_quantile: float = 0.05
    rating_col: Optional[str] = None  # auto-detect if None
    timestamp_col: Optional[str] = None  # auto-detect if None
    top_n_co_missing: int = 10
    output_dir: str = ""  # Empty string = derived by RunContext
    report_path: str = ""  # Empty string = derived by RunContext
    json_path: Optional[str] = None
    json_only: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)
