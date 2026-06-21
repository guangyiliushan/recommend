"""Feature effectiveness analysis — single-feature AUC ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class EffectivenessResult:
    """Feature effectiveness analysis results."""

    feature_auc: Dict[str, float]  # column_name → AUC (0-1 range)
    skipped_features: Dict[str, str]  # column_name → skip reason
    skipped: bool = False
    skip_reason: Optional[str] = None


def analyze(
    df: pd.DataFrame,
    label_col: str = "label_type",
) -> EffectivenessResult:
    """Compute single-feature AUC for each numeric feature against the label.

    Features are skipped if:
        - They are non-numeric or cannot be coerced to numeric.
        - They are constant (single unique value).
        - They are fully NaN.
        - They have fewer than 2 distinct values (AUC undefined).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    label_col : str
        Column to use as binary label for AUC computation.

    Returns
    -------
    EffectivenessResult
    """
    if df.empty:
        return EffectivenessResult(
            feature_auc={},
            skipped_features={},
            skipped=True,
            skip_reason="DataFrame is empty.",
        )

    # Check label availability
    if label_col not in df.columns:
        return EffectivenessResult(
            feature_auc={},
            skipped_features={},
            skipped=True,
            skip_reason=f"Label column '{label_col}' not found in DataFrame.",
        )

    y = df[label_col].values
    # Check label has at least 2 classes
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return EffectivenessResult(
            feature_auc={},
            skipped_features={},
            skipped=True,
            skip_reason=(
                f"Label column '{label_col}' has only {len(unique_labels)} unique "
                "value(s), AUC requires at least 2 classes."
            ),
        )

    feature_auc: Dict[str, float] = {}
    skipped_features: Dict[str, str] = {}

    # Exclude label column and core ID columns from feature set
    skip_cols = {label_col, "user_id", "item_id", "label_time", "label_type"}
    candidate_cols = [c for c in df.columns if c not in skip_cols]

    for col in candidate_cols:
        # Try to convert to numeric
        try:
            x = pd.to_numeric(df[col], errors="coerce").values
        except (ValueError, TypeError):
            skipped_features[col] = "Cannot convert to numeric"
            continue

        # Handle NaN: use median imputation for simplicity
        nan_mask = np.isnan(x)
        if nan_mask.all():
            skipped_features[col] = "All values are NaN"
            continue

        if nan_mask.any():
            median_val = np.nanmedian(x)
            x = np.where(nan_mask, median_val, x)

        # Check constant
        unique_x = np.unique(x)
        if len(unique_x) < 2:
            skipped_features[col] = (
                f"Constant value ({len(unique_x)} unique value(s))"
            )
            continue

        # Compute AUC
        try:
            if len(unique_labels) == 2:
                # Binary classification → standard AUC
                auc = float(roc_auc_score(y, x))
            else:
                # Multi-class → macro-average one-vs-rest AUC
                auc = float(roc_auc_score(y, x, multi_class="ovr", average="macro"))
        except ValueError as e:
            skipped_features[col] = f"AUC computation failed: {e}"
            continue

        feature_auc[col] = round(auc, 4)

    logger.info(
        "Effectiveness: %d features scored, %d skipped.",
        len(feature_auc),
        len(skipped_features),
    )

    return EffectivenessResult(
        feature_auc=feature_auc,
        skipped_features=skipped_features,
    )
