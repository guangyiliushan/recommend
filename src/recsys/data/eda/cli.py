"""EDA module CLI 鈥?command-line entry point for dataset exploratory data analysis.

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
    analyze_sequence,
    analyze_user_item,
)

logger = logging.getLogger(__name__)

# ---- Profile → Stats module mapping ----
_PROFILE_STATS_MAP = {
    "behavior": ["overview", "missing", "distribution", "sequence", "user_item", "effectiveness"],
    "feature": ["overview", "missing", "distribution"],
    "candidate": ["overview", "user_item"],
    "vector": ["overview", "vector"],
}

# ---- Profile → auto-subset mapping ----
_SUBSET_TO_PROFILE = {
    "seq": "behavior",
    "user_feat": "feature",
    "item_feat": "feature",
    "candidate": "candidate",
}
# mm_emb_* → vector (prefix match)


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


def _infer_profile(subset: str) -> str:
    """Infer analysis profile from subset name.

    - Explicit mapping for seq/user_feat/item_feat/candidate
    - mm_emb_* → "vector"
    - Fallback: "behavior"
    """
    if subset in _SUBSET_TO_PROFILE:
        return _SUBSET_TO_PROFILE[subset]
    if subset.startswith("mm_emb_"):
        return "vector"
    return "behavior"


def _run_profile(
    df: pd.DataFrame,
    profile: str,
    config: "EDAConfig",
    vector_store: Any = None,
) -> Dict[str, Any]:
    """Run stats modules selectively based on profile.

    Parameters
    ----------
    df : pd.DataFrame
        Tabular data for non-vector analysis.
    profile : str
        One of "behavior", "feature", "candidate", "vector".
    config : EDAConfig
    vector_store : VectorStore, optional
        Required for "vector" profile.

    Returns
    -------
    Dict mapping stat name → result dataclass.
    """
    active = _PROFILE_STATS_MAP.get(profile, _PROFILE_STATS_MAP["behavior"])
    results: Dict[str, Any] = {}

    if "overview" in active:
        results["overview"] = analyze_overview(df, label_col=config.label_col)
    if "missing" in active:
        results["missing"] = analyze_missing(
            df, label_col=config.label_col, top_n_co_missing=config.top_n_co_missing,
        )
    if "distribution" in active:
        results["distribution"] = analyze_distribution(
            df, label_col=config.label_col, dense_pattern=config.dense_pattern,
        )
    if "sequence" in active:
        results["sequence"] = analyze_sequence(df, domain_pattern=config.domain_pattern)
    if "effectiveness" in active:
        results["effectiveness"] = analyze_effectiveness(df, label_col=config.label_col)
    if "user_item" in active:
        results["user_item"] = analyze_user_item(
            df, user_col=config.user_col, item_col=config.item_col,
            domain_pattern=config.domain_pattern,
        )
    if "vector" in active:
        from recsys.data.eda.stats.vector import analyze as analyze_vector

        if vector_store is not None:
            results["vector"] = analyze_vector(vector_store)
        else:
            from recsys.data.eda.stats.vector import VectorResult
            results["vector"] = VectorResult(
                dim=0, n_vectors=0, original_count=0, sampled_at_load=False,
                norm_stats={}, zero_vector_count=0, zero_vector_ratio=0.0,
                duplicate_ratio=0.0, dim_variance={}, modality="unknown",
                skipped=True, skip_reason="No vector data provided.",
            )

    return results


def aggregate_reports(
    dataset_id: str,
    analysis_dir: str = "docs/analysis/dataset-eda",
) -> Path:
    """Aggregate per-subset summary.json files into a dataset-level index.md.

    Reads only JSON — never loads raw data.
    """
    import json
    from pathlib import Path

    base = Path(analysis_dir) / dataset_id
    if not base.exists():
        raise FileNotFoundError(f"Analysis directory not found: {base}")

    summaries = sorted(base.glob("*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No summary.json files found under {base}")

    lines = [f"# {dataset_id} 数据集分析总览\n"]
    lines.append("> 本页面由 `recsys-dataset-eda aggregate` 自动生成。\n")
    lines.append(f"## 分析子集 ({len(summaries)})\n")

    overview_entries: list[tuple[str, dict]] = []

    for summary_path in summaries:
        subset_name = summary_path.parent.name
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        ov = data.get("overview", {})
        total_rows = ov.get("total_rows", 0)
        total_cols = ov.get("total_columns", 0)
        skipped = ov.get("skipped", False)

        report_rel = Path(subset_name) / "report.md"
        status = " (skipped)" if skipped else ""
        lines.append(f"- [{subset_name}{status}]({report_rel}) — {total_rows:,} rows, {total_cols} cols")

        if not skipped:
            overview_entries.append((subset_name, ov))

    # Key metrics comparison table (if any subset had overview)
    if overview_entries:
        lines.append("\n## 关键指标对比\n")
        lines.append("| Subset | Rows | Columns |")
        lines.append("|--------|------|---------|")
        for name, ov in overview_entries:
            lines.append(f"| {name} | {ov.get('total_rows', 0):,} | {ov.get('total_columns', 0)} |")
        lines.append("")

        # Per-subset row count bar summary
        lines.append("## 子集规模分布\n")
        for name, ov in overview_entries:
            rows = ov.get("total_rows", 0)
            bar = "█" * min(50, max(1, rows // 10000))
            lines.append(f"- **{name}**: {rows:,} rows {bar}")
        lines.append("")

    lines.append("---\n")
    lines.append("*本页面由 `recsys.data.eda.cli.aggregate_reports` 生成。*\n")

    index_path = base / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Aggregated %d subsets → %s", len(summaries), index_path)
    return index_path

def _build_stats_json(
    overview: Any,
    missing: Any,
    distribution: Any,
    sequence: Any,
    effectiveness: Any,
    user_item: Any,
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

    # ---- 3. Stats JSON (if requested) ----
    stats_json = _build_stats_json(
        overview, missing, distribution, sequence, effectiveness,
        user_item, sample_meta,
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


def _run_eda_subset(
    ds_instance,
    args,
    subset: str,
    profile: str,
    ctx,
) -> Dict[str, Any]:
    """Run EDA for a single subset using the dataset adapter's load_subset()."""
    from recsys.data.eda.sampler import SampleMetadata

    # Load single subset (memory-safe)
    data, load_meta = ds_instance.load_subset(
        subset, max_rows=args.max_rows, seed=args.sample_seed
    )

    from recsys.data.datasets.taac2025 import VectorStore

    vector_store: Any = None

    if isinstance(data, VectorStore):
        # Vector subset — use to_dataframe for overview
        vector_store = data
        df = data.to_dataframe()
        original_rows = load_meta["original_rows"] if load_meta else data.original_count
    else:
        df = data
        original_rows = load_meta["original_rows"] if load_meta else len(df)

    # Build metadata
    temp_metadata = SampleMetadata(
        sample_strategy="pending",
        total_rows=original_rows,
        sample_ratio=1.0,
        strat_rows=0,
        tail_rows=0,
        union_rows=len(df),
        seed=args.sample_seed,
    )

    ctx.subset = subset
    ctx.profile = profile
    ctx.sample_metadata = temp_metadata

    # Re-derive paths with subset in RunContext
    ctx._apply_subset_paths()

    # Run profile-selected stats
    stats_results = _run_profile(df, profile, EDAConfig(
        max_rows=args.max_rows,
        sample_seed=args.sample_seed,
        output_dir=str(ctx.output_dir),
        report_path=str(ctx.report_path),
        json_path=str(ctx.json_path) if ctx.json_path else None,
        json_only=args.json_only,
    ), vector_store=vector_store)

    # Build stats JSON
    stats_json = _build_stats_json(
        stats_results.get("overview"),
        stats_results.get("missing"),
        stats_results.get("distribution"),
        stats_results.get("sequence"),
        stats_results.get("effectiveness"),
        stats_results.get("user_item"),
        temp_metadata,
    )
    # Add vector stats if present
    if "vector" in stats_results:
        from dataclasses import asdict
        stats_json["vector"] = asdict(stats_results["vector"])

    # JSON-only mode
    if args.json_only:
        ctx.json_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.json_path.write_text(json.dumps(stats_json, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("JSON-only stats written to %s", ctx.json_path)
        return {"status": "ok", "mode": "json_only", "json_path": str(ctx.json_path)}

    # Save JSON
    if ctx.json_path:
        ctx.json_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.json_path.write_text(json.dumps(stats_json, indent=2, ensure_ascii=False), encoding="utf-8")

    # Render ECharts
    render_out = render_to_echarts(
        overview=stats_results.get("overview"),
        missing=stats_results.get("missing"),
        distribution=stats_results.get("distribution"),
        sequence=stats_results.get("sequence"),
        effectiveness=stats_results.get("effectiveness"),
        user_item=stats_results.get("user_item"),
        metadata=temp_metadata,
        output_dir=ctx.output_dir,
        ctx=ctx,
        vector_result=stats_results.get("vector"),
    )

    # Generate report
    generate_markdown_report(
        overview=stats_results.get("overview"),
        missing=stats_results.get("missing"),
        distribution=stats_results.get("distribution"),
        sequence=stats_results.get("sequence"),
        effectiveness=stats_results.get("effectiveness"),
        user_item=stats_results.get("user_item"),
        metadata=temp_metadata,
        chart_files=render_out.chart_files,
        output_path=ctx.report_path,
        ctx=ctx,
    )

    return {
        "status": "ok",
        "mode": "full",
        "chart_count": render_out.chart_count,
        "chart_dir": str(ctx.output_dir),
        "report_path": str(ctx.report_path),
    }


def _cmd_analyze(args, parser) -> int:
    """Handle 'analyze' subcommand."""
    from recsys.data.eda.context import RunContext
    from recsys.data.eda.sampler import SampleMetadata

    # Load dataset
    if args.dataset is None:
        print("ERROR: --dataset is required for 'analyze'.", file=sys.stderr)
        return 1

    try:
        from recsys.core.registry import DATASET_REGISTRY
    except ImportError:
        print("ERROR: Cannot import DATASET_REGISTRY.", file=sys.stderr)
        return 1

    ds_cls = DATASET_REGISTRY.get(args.dataset)
    if ds_cls is None:
        print(f"ERROR: Dataset '{args.dataset}' not found.", file=sys.stderr)
        return 1

    ds_instance = ds_cls()

    # Determine subsets to analyze
    raw = ds_instance._load_raw()
    manifest = raw.get("manifest")

    if manifest is not None and args.all_subsets:
        subsets_to_run = manifest.list_subsets()
    elif manifest is not None and args.subset == "auto":
        subsets_to_run = [manifest.default_eda_subset]
    elif manifest is not None:
        subsets_to_run = [args.subset]
    else:
        # No manifest — old-style dataset, fall through to backward compat
        subsets_to_run = ["__single__"]

    if not args.all_subsets and len(subsets_to_run) > 1:
        logger.info("Multiple subsets available: %s. Use --all-subsets to run all.", subsets_to_run)

    all_ok = True
    for subset in subsets_to_run:
        if manifest is not None:
            profile = args.profile or manifest.auto_profile(subset)
        else:
            profile = args.profile or "behavior"
            subset = args.subset

        logger.info("=== Analyzing subset '%s' (profile=%s) ===", subset, profile)

        # Build context with subset paths
        ctx = RunContext.from_args(
            dataset=args.dataset,
            dataset_path=None,
            dataset_id=args.dataset_id,
            run_tag=args.run_tag,
            sample_metadata=SampleMetadata(
                sample_strategy="pending", total_rows=0, sample_ratio=1.0,
                strat_rows=0, tail_rows=0, union_rows=0, seed=args.sample_seed,
            ),
            output_dir=args.output_dir,
            report_path=args.report_path,
            json_path=args.json_path,
            load_sampled=False,
            load_original_rows=0,
        )
        ctx.subset = subset
        ctx.profile = profile
        ctx._apply_subset_paths()

        try:
            result = _run_eda_subset(ds_instance, args, subset, profile, ctx)
            if result["status"] != "ok":
                all_ok = False
        except Exception as e:
            logger.error("Failed to analyze subset '%s': %s", subset, e, exc_info=args.verbose)
            all_ok = False

    if not args.json_only and not args.no_index_update:
        try:
            from recsys.data.eda.index import update_index_page
            update_index_page()
        except Exception as e:
            logger.warning("Failed to update index: %s", e)

    return 0 if all_ok else 1


def _cmd_aggregate(args, parser) -> int:
    """Handle 'aggregate' subcommand."""
    try:
        idx_path = aggregate_reports(args.dataset)
        print(f"Aggregated overview: {idx_path}")

        # Also update global index
        from recsys.data.eda.index import update_index_page
        update_index_page()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for dataset EDA.

    Subcommands (P1):
        recsys-dataset-eda analyze --dataset taac2025_1M --subset seq
        recsys-dataset-eda aggregate --dataset taac2025_1M

    Backward compatible (no subcommand):
        recsys-dataset-eda --dataset taac2026_data_sample
        recsys-dataset-eda --dataset-path data.csv
    """
    # Detect subcommand vs backward-compat mode
    if argv is None:
        argv = sys.argv[1:]

    subcommand = None
    sub_args = list(argv)
    for i, arg in enumerate(argv):
        if arg in ("analyze", "aggregate"):
            subcommand = arg
            sub_args = argv[i + 1:]
            break

    if subcommand is None:
        # Backward-compatible mode (original behavior)
        return _main_backward(sub_args)

    if subcommand == "aggregate":
        parser = argparse.ArgumentParser(description="Aggregate subset EDA reports.")
        parser.add_argument("--dataset", required=True, help="Dataset ID (e.g. taac2025_1M).")
        parser.add_argument("--verbose", action="store_true")
        args = parser.parse_args(sub_args)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return _cmd_aggregate(args, parser)

    if subcommand == "analyze":
        parser = argparse.ArgumentParser(description="Analyze a single dataset subset.")
        parser.add_argument("--dataset", required=True, help="Dataset ID (e.g. taac2025_1M).")
        parser.add_argument("--subset", default="auto", help="Subset name (seq/user_feat/item_feat/candidate/mm_emb_text/...).")
        parser.add_argument("--profile", default=None, help="Analysis profile (behavior/feature/candidate/vector). Auto-inferred if not set.")
        parser.add_argument("--all-subsets", action="store_true", help="Run all subsets sequentially.")
        parser.add_argument("--dataset-id", default=None, help="Explicit dataset identifier override.")
        parser.add_argument("--run-tag", default=None, help="Optional version tag.")
        parser.add_argument("--max-rows", type=int, default=500_000, help="Maximum rows after sampling.")
        parser.add_argument("--sample-seed", type=int, default=42, help="Random seed.")
        parser.add_argument("--output-dir", default=None, help="ECharts output directory.")
        parser.add_argument("--report-path", default=None, help="Markdown report path.")
        parser.add_argument("--json-path", default=None, help="Stats JSON path.")
        parser.add_argument("--json-only", action="store_true", help="Only output JSON.")
        parser.add_argument("--no-index-update", action="store_true", help="Skip index update.")
        parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
        args = parser.parse_args(sub_args)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return _cmd_analyze(args, parser)

    return 0


def _main_backward(argv: Sequence[str]) -> int:
    """Backward-compatible CLI path (no subcommand) — original behavior."""
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


if __name__ == "__main__":
    raise SystemExit(main())



