"""Stats layer — pure statistical computation on pandas DataFrames.

Each module exports:
    - A result dataclass (e.g. OverviewResult)
    - An analyze() function with signature analyze(df: pd.DataFrame, **kw) -> Result

All result dataclasses include:
    - skipped: bool = False
    - skip_reason: Optional[str] = None
"""

# ruff: noqa: F401

from recsys.data.eda.stats.distribution import (  # noqa: E501
    DistributionResult,
)
from recsys.data.eda.stats.distribution import (
    analyze as analyze_distribution,
)
from recsys.data.eda.stats.effectiveness import (  # noqa: E501
    EffectivenessResult,
)
from recsys.data.eda.stats.effectiveness import (
    analyze as analyze_effectiveness,
)
from recsys.data.eda.stats.missing import (
    MissingResult,
)
from recsys.data.eda.stats.missing import (
    analyze as analyze_missing,
)
from recsys.data.eda.stats.overview import (
    OverviewResult,
)
from recsys.data.eda.stats.overview import (
    analyze as analyze_overview,
)
from recsys.data.eda.stats.sequence import (
    SequenceResult,
)
from recsys.data.eda.stats.sequence import (
    analyze as analyze_sequence,
)
from recsys.data.eda.stats.user_item import (
    UserItemResult,
)
from recsys.data.eda.stats.user_item import (
    analyze as analyze_user_item,
)
