"""Stats layer — pure statistical computation on pandas DataFrames.

Each module exports:
    - A result dataclass (e.g. OverviewResult)
    - An analyze() function with signature analyze(df: pd.DataFrame, **kw) -> Result

All result dataclasses include:
    - skipped: bool = False
    - skip_reason: Optional[str] = None
"""

from recsys.data.eda.stats.distribution import DistributionResult
from recsys.data.eda.stats.distribution import analyze as analyze_distribution
from recsys.data.eda.stats.effectiveness import EffectivenessResult
from recsys.data.eda.stats.effectiveness import analyze as analyze_effectiveness
from recsys.data.eda.stats.missing import MissingResult
from recsys.data.eda.stats.missing import analyze as analyze_missing
from recsys.data.eda.stats.overview import OverviewResult
from recsys.data.eda.stats.overview import analyze as analyze_overview
from recsys.data.eda.stats.rating import RatingResult
from recsys.data.eda.stats.rating import analyze as analyze_rating
from recsys.data.eda.stats.sequence import SequenceResult
from recsys.data.eda.stats.sequence import analyze as analyze_sequence
from recsys.data.eda.stats.sparsity import SparsityResult
from recsys.data.eda.stats.sparsity import analyze as analyze_sparsity
from recsys.data.eda.stats.temporal import TemporalResult
from recsys.data.eda.stats.temporal import analyze as analyze_temporal
from recsys.data.eda.stats.user_item import UserItemResult
from recsys.data.eda.stats.user_item import analyze as analyze_user_item
from recsys.data.eda.stats.vector import VectorResult
from recsys.data.eda.stats.vector import analyze as analyze_vector

__all__ = [
    "DistributionResult",
    "analyze_distribution",
    "EffectivenessResult",
    "analyze_effectiveness",
    "MissingResult",
    "analyze_missing",
    "OverviewResult",
    "analyze_overview",
    "RatingResult",
    "analyze_rating",
    "SequenceResult",
    "analyze_sequence",
    "SparsityResult",
    "analyze_sparsity",
    "TemporalResult",
    "analyze_temporal",
    "UserItemResult",
    "analyze_user_item",
    "VectorResult",
    "analyze_vector",
]
