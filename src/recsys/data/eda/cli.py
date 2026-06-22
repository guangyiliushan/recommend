"""EDA module CLI — command-line entry point for dataset exploratory data analysis.

Entry point registered in pyproject.toml:
    recsys-dataset-eda = "recsys.data.eda.cli:main"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from recsys.data.eda import EDAConfig, SampleMetadata, hybrid_sample
from recsys.data.eda.render import render_to_echarts
from recsys.data.eda.report import generate_markdown_report
from recsys.data.eda.stats import (
    analyze_distribution,
    analyze_effectiveness,
    analyze_missing,
    analyze_overview,
    analyze_rating,
    analyze_sequence,
    analyze_sparsity,
    analyze_temporal,
    analyze_user_item,
)

logger = logging.getLogger(__name__)


def _load_dataframe(dataset_path: str) -> pd.DataFrame:
    """Load a DataFrame from a local file path.

    Supports parquet (.parquet), feather (.feather, .ipc), and CSV (.csv).
    """
    path = Path(dataset_path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix in (".feather", ".ipc"):
        return pd.read_feather(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported: .parquet, .feather, .csv"
        )


def _extract_dataframe(
    raw: dict,
    max_rows: int = 500_000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Extract a pandas DataFrame from _load_raw() result dict.

    For HuggingFace Dataset / pyarrow Table sources, pre-samples at the arrow
    level (O(1) row count, O(k) row selection) to avoid loading 100GB+ into
    memory before the main sampling pipeline runs.

    Returns (df, load_meta) where load_meta may contain:
        {"original_rows": N, "sampled_at_load": True}
    or None if no pre-sampling was needed.

    Tries known keys (dataset, seq, train) in order; falls back to the first
    value that supports to_pandas(), or tries pd.DataFrame() as last resort.
    """
    rng = np.random.default_rng(seed)
    load_meta: Optional[Dict[str, Any]] = None

    # Try known key patterns
    for key in ("dataset", "seq", "train"):
        if key not in raw:
            continue
        val = raw[key]

        # Case 1: HuggingFace Dataset — pre-sample with .select()
        if hasattr(val, "select") and hasattr(val, "to_pandas"):
            total = len(val)
            if total > max_rows:
                indices = sorted(
                    rng.choice(total, max_rows, replace=False).tolist()
                )
                val = val.select(indices)
                load_meta = {"original_rows": total, "sampled_at_load": True}
                logger.info(
                    "Pre-sampled HuggingFace Dataset: %d → %d rows.",
                    total,
                    max_rows,
                )
            return val.to_pandas(), load_meta

        # Case 2: pyarrow Table — pre-sample with .take()
        if hasattr(val, "take") and hasattr(val, "to_pandas"):
            total = len(val)
            if total > max_rows:
                import pyarrow as pa
                idx_array = pa.array(
                    rng.choice(total, max_rows, replace=False)
                )
                val = val.take(idx_array)
                load_meta = {"original_rows": total, "sampled_at_load": True}
                logger.info(
                    "Pre-sampled pyarrow Table: %d → %d rows.",
                    total,
                    max_rows,
                )
            return val.to_pandas(), load_meta

        # Case 3: Already a DataFrame — pass through
        if isinstance(val, pd.DataFrame):
            return val, None

        # Case 4: List of lists (e.g. MovieLens JSON: [[item1, item2], ...])
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
            records = []
            for user_idx, items in enumerate(val):
                for item_id in items:
                    records.append({"user_id": user_idx, "item_id": int(item_id)})
            df = pd.DataFrame(records)
            if len(df) > max_rows:
                indices = sorted(
                    rng.choice(len(df), max_rows, replace=False).tolist()
                )
                df = df.iloc[indices].reset_index(drop=True)
                load_meta = {"original_rows": len(records), "sampled_at_load": True}
                logger.info(
                    "Pre-sampled list-of-lists: %d → %d rows.",
                    len(records), max_rows,
                )
            return df, load_meta

    # Try any value from the dict as last resort
    for val in raw.values():
        if hasattr(val, "select") and hasattr(val, "to_pandas"):
            total = len(val)
            if total > max_rows:
                indices = sorted(
                    rng.choice(total, max_rows, replace=False).tolist()
                )
                val = val.select(indices)
                load_meta = {"original_rows": total, "sampled_at_load": True}
            return val.to_pandas(), load_meta
        if hasattr(val, "take") and hasattr(val, "to_pandas"):
            total = len(val)
            if total > max_rows:
                import pyarrow as pa
                idx_array = pa.array(
                    rng.choice(total, max_rows, replace=False)
                )
                val = val.take(idx_array)
                load_meta = {"original_rows": total, "sampled_at_load": True}
            return val.to_pandas(), load_meta
        if hasattr(val, "to_pandas"):
            return val.to_pandas(), None
        # Case 4 (last resort): List of lists
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
            records = []
            for user_idx, items in enumerate(val):
                for item_id in items:
                    records.append({"user_id": user_idx, "item_id": int(item_id)})
            df = pd.DataFrame(records)
            if len(df) > max_rows:
                indices = sorted(
                    rng.choice(len(df), max_rows, replace=False).tolist()
                )
                df = df.iloc[indices].reset_index(drop=True)
                load_meta = {"original_rows": len(records), "sampled_at_load": True}
                logger.info(
                    "Pre-sampled list-of-lists (last resort): %d → %d rows.",
                    len(records), max_rows,
                )
            return df, load_meta
        if isinstance(val, pd.DataFrame):
            return val, None

    # Last resort
    raise KeyError(
        f"Cannot extract DataFrame from _load_raw() dict. "
        f"Available keys: {list(raw.keys())}"
    )


def _build_stats_json(
    overview: Any,
    missing: Any,
    distribution: Any,
    sequence: Any,
    effectiveness: Any,
    user_item: Any,
    sparsity: Any,
    temporal: Any,
    rating: Any,
    metadata: SampleMetadata,
) -> Dict[str, Any]:
    """Build a structured JSON of all stats (for --json-only or --json-path mode)."""
    from dataclasses import asdict

    return {
        "overview": asdict(overview),
        "missing": asdict(missing),
        "distribution": asdict(distribution),
        "sequence": asdict(sequence),
        "effectiveness": asdict(effectiveness),
        "user_item": asdict(user_item),
        "sparsity": asdict(sparsity),
        "temporal": asdict(temporal),
        "rating": asdict(rating),
        "sample_metadata": {
            "sample_strategy": metadata.sample_strategy,
            "total_rows": metadata.total_rows,
            "sample_ratio": metadata.sample_ratio,
            "strat_rows": metadata.strat_rows,
            "tail_rows": metadata.tail_rows,
            "union_rows": metadata.union_rows,
            "seed": metadata.seed,
            "total_users": metadata.total_users,
            "total_items": metadata.total_items,
        },
    }


def run_eda(config: EDAConfig, df: Optional[pd.DataFrame] = None, ctx: Any = None) -> Dict[str, Any]:
    """Run the full EDA pipeline.

    Parameters
    ----------
    config : EDAConfig
        Pipeline configuration.
    df : pd.DataFrame, optional
        Pre-loaded DataFrame. If None, will be loaded from CLI args.
    ctx : Any, optional
        RunContext for enriched metadata and dynamic paths.

    Returns
    -------
    Dict[str, Any]
        Pipeline results including stats and output paths.
    """
    if df is None or df.empty:
        return {"status": "error", "message": "No data provided."}

    # ---- 1. Sample ----
    df_sampled, sample_meta = hybrid_sample(
        df,
        max_rows=config.max_rows,
        label_col=config.label_col,
        item_col=config.item_col,
        user_col=config.user_col,
        seed=config.sample_seed,
        tail_quantile=config.tail_quantile,
    )

    # ---- 2. Stats ----
    overview = analyze_overview(df_sampled, label_col=config.label_col)
    missing = analyze_missing(
        df_sampled,
        label_col=config.label_col,
        top_n_co_missing=config.top_n_co_missing,
    )
    distribution = analyze_distribution(
        df_sampled,
        label_col=config.label_col,
        dense_pattern=config.dense_pattern,
    )
    sequence = analyze_sequence(df_sampled, domain_pattern=config.domain_pattern)
    effectiveness = analyze_effectiveness(df_sampled, label_col=config.label_col)
    user_item = analyze_user_item(
        df_sampled,
        user_col=config.user_col,
        item_col=config.item_col,
        domain_pattern=config.domain_pattern,
    )
    sparsity = analyze_sparsity(
        df_sampled,
        user_col=config.user_col,
        item_col=config.item_col,
        cold_start_quantile=getattr(config, "cold_start_quantile", 0.05),
    )
    temporal = analyze_temporal(
        df_sampled,
        user_col=config.user_col,
        timestamp_col=getattr(config, "timestamp_col", None) or None,
    )
    rating = analyze_rating(
        df_sampled,
        user_col=config.user_col,
        item_col=config.item_col,
        rating_col=getattr(config, "rating_col", None) or None,
    )

    # ---- 3. Stats JSON (if requested) ----
    stats_json = _build_stats_json(
        overview, missing, distribution, sequence, effectiveness,
        user_item, sparsity, temporal, rating, sample_meta,
    )

    # ---- 4. JSON-only mode ----
    if config.json_only:
        json_path = config.json_path or "stats.json"
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats_json, f, indent=2, ensure_ascii=False)
        logger.info("JSON-only stats written to %s", json_path)
        return {
            "status": "ok",
            "mode": "json_only",
            "json_path": json_path,
            "stats": stats_json,
        }

    # Save stats JSON if --json-path specified (even in full mode)
    if config.json_path:
        Path(config.json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config.json_path, "w", encoding="utf-8") as f:
            json.dump(stats_json, f, indent=2, ensure_ascii=False)

    # ---- 5. Render ECharts ----
    output_dir = Path(config.output_dir)
    render_out = render_to_echarts(
        overview=overview,
        missing=missing,
        distribution=distribution,
        sequence=sequence,
        effectiveness=effectiveness,
        user_item=user_item,
        sparsity=sparsity,
        temporal=temporal,
        rating=rating,
        metadata=sample_meta,
        output_dir=output_dir,
        ctx=ctx,
    )

    # ---- 6. Generate Markdown report ----
    report_path = Path(config.report_path)
    generate_markdown_report(
        overview=overview,
        missing=missing,
        distribution=distribution,
        sequence=sequence,
        effectiveness=effectiveness,
        user_item=user_item,
        sparsity=sparsity,
        temporal=temporal,
        rating=rating,
        metadata=sample_meta,
        chart_files=render_out.chart_files,
        output_path=report_path,
        ctx=ctx,
    )

    logger.info(
        "EDA complete: %d charts, report at %s",
        render_out.chart_count,
        report_path,
    )

    return {
        "status": "ok",
        "mode": "full",
        "chart_count": render_out.chart_count,
        "chart_dir": str(output_dir),
        "report_path": str(report_path),
        "stats": stats_json,
    }


def _main_backward(argv: Sequence[str]) -> int:
    """Backward-compatible CLI path — original behavior."""
    parser = argparse.ArgumentParser(
        description="Dataset Exploratory Data Analysis (EDA) tool for RecBench.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Registered dataset name (e.g. taac2026_data_sample).",
    )
    parser.add_argument(
        "--dataset-path", default=None,
        help="Path to a local parquet/feather/csv file.",
    )
    parser.add_argument(
        "--dataset-id", default=None,
        help="Explicit dataset identifier.",
    )
    parser.add_argument(
        "--run-tag", default=None,
        help="Optional version tag for multi-run retention.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=500_000,
        help="Maximum rows after sampling.",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--rating-col", default=None,
        help="Rating column name (auto-detect if None).",
    )
    parser.add_argument(
        "--timestamp-col", default=None,
        help="Timestamp column name (auto-detect if None).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for ECharts JSON output.",
    )
    parser.add_argument(
        "--report-path", default=None,
        help="Path for the Markdown report.",
    )
    parser.add_argument(
        "--json-path", default=None,
        help="Path for structured stats JSON output.",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Only output structured stats JSON.",
    )
    parser.add_argument(
        "--no-index-update", action="store_true",
        help="Disable automatic update of docs/analysis/index.md.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- Load data ----
    if args.dataset is not None:
        try:
            from recsys.core.registry import DATASET_REGISTRY
        except ImportError:
            print("ERROR: Cannot import DATASET_REGISTRY.", file=sys.stderr)
            return 1

        ds_cls = DATASET_REGISTRY.get(args.dataset)
        if ds_cls is None:
            print(f"ERROR: Dataset '{args.dataset}' not found.", file=sys.stderr)
            return 1

        try:
            ds_instance = ds_cls()
            raw = ds_instance._load_raw()
            df, load_meta = _extract_dataframe(
                raw, max_rows=args.max_rows, seed=args.sample_seed
            )
            logger.info("Loaded dataset '%s': %d rows.", args.dataset, len(df))
        except Exception as e:
            print(f"ERROR: Failed to load dataset '{args.dataset}': {e}", file=sys.stderr)
            return 1

    elif args.dataset_path is not None:
        try:
            df = _load_dataframe(args.dataset_path)
            load_meta = None
            logger.info("Loaded file '%s': %d rows.", args.dataset_path, len(df))
        except Exception as e:
            print(f"ERROR: Failed to load file '{args.dataset_path}': {e}", file=sys.stderr)
            return 1
    else:
        print("ERROR: Either --dataset or --dataset-path must be specified.", file=sys.stderr)
        return 1

    from recsys.data.eda.context import RunContext
    from recsys.data.eda.sampler import SampleMetadata

    original_rows = load_meta["original_rows"] if load_meta else len(df)
    load_sampled = bool(load_meta and load_meta.get("sampled_at_load"))
    temp_metadata = SampleMetadata(
        sample_strategy="pending",
        total_rows=original_rows,
        sample_ratio=1.0,
        strat_rows=0,
        tail_rows=0,
        union_rows=len(df),
        seed=args.sample_seed,
    )

    ctx = RunContext.from_args(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        dataset_id=args.dataset_id,
        run_tag=args.run_tag,
        sample_metadata=temp_metadata,
        output_dir=args.output_dir,
        report_path=args.report_path,
        json_path=args.json_path,
        load_sampled=load_sampled,
        load_original_rows=original_rows,
    )

    logger.info("Dataset ID: %s | Output: %s", ctx.dataset_id, ctx.output_dir)

    config = EDAConfig(
        max_rows=args.max_rows,
        sample_seed=args.sample_seed,
        output_dir=str(ctx.output_dir),
        report_path=str(ctx.report_path),
        json_path=str(ctx.json_path) if ctx.json_path else None,
        json_only=args.json_only,
        rating_col=getattr(args, "rating_col", None) or None,
        timestamp_col=getattr(args, "timestamp_col", None) or None,
    )

    result = run_eda(config, df, ctx=ctx)

    if result["status"] == "error":
        print(f"ERROR: {result.get('message', 'Unknown error')}", file=sys.stderr)
        return 1

    if not args.json_only and not args.no_index_update:
        try:
            from recsys.data.eda.index import update_index_page
            update_index_page()
        except Exception as e:
            logger.warning("Failed to update index: %s", e)

    if args.json_only:
        print(f"Stats JSON written to {result['json_path']}")
    else:
        print(f"Charts ({result['chart_count']}): {result['chart_dir']}")
        print(f"Report: {result['report_path']}")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    return _main_backward(argv)


if __name__ == "__main__":
    raise SystemExit(main())
