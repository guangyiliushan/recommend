"""Synthetic benchmark: compare storage formats, compression codecs, and backends.

Generates synthetic recommendation-style data, then measures:
    - Read time (wall-clock)
    - Write/materialize time
    - Repeat-run cache hit time
    - Output file size & compression ratio
    - Peak memory usage (RSS where available)

Outputs:
    outputs/data-benchmarks/{run_id}/
        summary.csv     – per-run summary
        formats.csv     – format × compression comparison
        backends.csv    – backend comparison
        memory.csv      – memory usage per run
        report.md       – human-readable report
        report.json     – machine-readable report

Usage:
    uv run python scripts/benchmark_data_pipeline.py --rows 1000000
    uv run python scripts/benchmark_data_pipeline.py --rows 10000000 --formats parquet feather --compressions snappy zstd
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from recsys.data.preprocessor import (
    CompressionCodec,
    ExecutionBackend,
    PreprocessingConfig,
    ResourceLimits,
    StorageConfig,
    StorageFormat,
    is_backend_available,
    materialize_to_columnar,
)


def _generate_synthetic_data(
    n_rows: int,
    n_users: int = 10_000,
    n_items: int = 50_000,
    n_feature_cols: int = 10,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> str:
    """Generate a synthetic recommendation dataset as CSV."""
    rng = np.random.default_rng(seed)

    user_ids = rng.integers(1, n_users + 1, size=n_rows)
    item_ids = rng.integers(1, n_items + 1, size=n_rows)
    labels = rng.integers(0, 2, size=n_rows).astype(np.float32)

    data: Dict[str, np.ndarray] = {
        "user_id": user_ids,
        "item_id": item_ids,
        "label": labels,
    }

    # Add feature columns (mix of numeric and categorical)
    for i in range(n_feature_cols):
        if i % 3 == 0:
            # Categorical-like
            data[f"feat_cat_{i}"] = rng.integers(1, min(100, n_items // 100) + 1, size=n_rows)
        elif i % 3 == 1:
            # Float
            data[f"feat_float_{i}"] = rng.normal(0, 1, size=n_rows).astype(np.float32)
        else:
            # Integer
            data[f"feat_int_{i}"] = rng.integers(0, 10_000_000, size=n_rows).astype(np.int64)

    df = pd.DataFrame(data)

    if output_path is None:
        output_dir = Path("./outputs/data-benchmarks/_synthetic")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"synth_{n_rows}.csv")

    df.to_csv(output_path, index=False)
    print(f"Generated {n_rows} rows → {output_path} ({Path(output_path).stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    source: str
    format: str
    compression: str
    backend: str
    n_rows: int
    read_time_s: float = 0.0
    materialize_time_s: float = 0.0
    output_size_mb: float = 0.0
    compression_ratio: float = 0.0
    cache_hit_time_s: Optional[float] = None
    cache_hit: bool = False
    error: Optional[str] = None


def _measure_memory_rss() -> int:
    """Get current process RSS in bytes."""
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return 0


def _run_single_benchmark(
    source_path: str,
    fmt: StorageFormat,
    comp: CompressionCodec,
    backend: ExecutionBackend,
) -> BenchmarkResult:
    """Run a single format × compression × backend benchmark."""
    n_rows = -1  # will be estimated from source

    # Count rows via quick estimate
    try:
        with open(source_path, "rb") as f:
            n_rows = sum(1 for _ in f) - 1
    except Exception:
        pass

    mem_before = _measure_memory_rss()

    config = PreprocessingConfig(
        source_path=source_path,
        backend=backend,
        storage=StorageConfig(
            format=fmt,
            compression=comp,
            row_group_size=128 * 1024,
            output_dir=f"./outputs/data-benchmarks/_cache/{fmt.value}_{comp.value}_{backend.value}",
        ),
        resources=ResourceLimits.auto(),
        downcast_int=True,
        downcast_float=True,
        auto_category=True,
    )

    t0 = time.perf_counter()
    try:
        artifact = materialize_to_columnar(source_path, config)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        return BenchmarkResult(
            source=source_path,
            format=fmt.value,
            compression=comp.value,
            backend=backend.value,
            n_rows=n_rows,
            error=str(e),
        )

    mem_after = _measure_memory_rss()
    _peak_mem_mb = (mem_after - mem_before) / (1024 * 1024) if mem_before > 0 else 0

    # Repeat run to measure cache hit
    t_cache = time.perf_counter()
    try:
        materialize_to_columnar(source_path, config)
        cache_elapsed = time.perf_counter() - t_cache
        cache_hit = True
    except Exception:
        cache_elapsed = None
        cache_hit = False

    return BenchmarkResult(
        source=source_path,
        format=fmt.value,
        compression=comp.value,
        backend=backend.value,
        n_rows=artifact.n_rows,
        read_time_s=elapsed,
        materialize_time_s=elapsed,  # read + write combined
        output_size_mb=artifact.file_size_mb,
        compression_ratio=artifact.compression_ratio,
        cache_hit_time_s=cache_elapsed,
        cache_hit=cache_hit,
    )


def run_benchmarks(
    source_path: str,
    formats: List[StorageFormat],
    compressions: List[CompressionCodec],
    backends: List[ExecutionBackend],
) -> List[BenchmarkResult]:
    """Run all benchmark combinations."""
    results: List[BenchmarkResult] = []
    total = len(formats) * len(compressions) * len(backends)
    idx = 0

    for fmt in formats:
        for comp in compressions:
            for backend in backends:
                idx += 1
                print(f"[{idx}/{total}] {fmt.value}/{comp.value}/{backend.value} ...", end=" ", flush=True)
                result = _run_single_benchmark(source_path, fmt, comp, backend)
                if result.error:
                    print(f"ERROR: {result.error}")
                else:
                    print(f"{result.read_time_s:.1f}s, {result.output_size_mb:.1f}MB")
                results.append(result)

    return results


def _write_reports(
    results: List[BenchmarkResult],
    output_dir: str,
    source_path: str,
) -> None:
    """Write all benchmark output files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    # summary.csv
    rows = []
    for r in results:
        rows.append({
            "format": r.format,
            "compression": r.compression,
            "backend": r.backend,
            "n_rows": r.n_rows,
            "read_time_s": round(r.read_time_s, 2),
            "output_size_mb": round(r.output_size_mb, 2),
            "compression_ratio": round(r.compression_ratio, 2),
            "cache_hit_time_s": round(r.cache_hit_time_s, 2) if r.cache_hit_time_s else "",
            "error": r.error or "",
        })
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)

    # formats.csv (aggregated by format × compression)
    fmt_rows = []
    for (fmt, comp), grp in pd.DataFrame(rows).groupby(["format", "compression"]):
        grp_ok = grp[grp["error"] == ""]
        fmt_rows.append({
            "format": fmt,
            "compression": comp,
            "n_runs": len(grp_ok),
            "mean_time_s": round(grp_ok["read_time_s"].mean(), 2) if len(grp_ok) > 0 else 0,
            "mean_size_mb": round(grp_ok["output_size_mb"].mean(), 2) if len(grp_ok) > 0 else 0,
            "mean_compression_ratio": round(grp_ok["compression_ratio"].mean(), 2) if len(grp_ok) > 0 else 0,
        })
    pd.DataFrame(fmt_rows).to_csv(out / "formats.csv", index=False)

    # backends.csv (aggregated by backend)
    be_rows = []
    for backend, grp in pd.DataFrame(rows).groupby("backend"):
        grp_ok = grp[grp["error"] == ""]
        be_rows.append({
            "backend": backend,
            "n_runs": len(grp_ok),
            "mean_time_s": round(grp_ok["read_time_s"].mean(), 2) if len(grp_ok) > 0 else 0,
            "n_errors": len(grp[grp["error"] != ""]),
        })
    pd.DataFrame(be_rows).to_csv(out / "backends.csv", index=False)

    # report.json
    report_json = {
        "generated_at": now,
        "source": source_path,
        "n_results": len(results),
        "results": [
            {
                "format": r.format,
                "compression": r.compression,
                "backend": r.backend,
                "read_time_s": round(r.read_time_s, 2),
                "output_size_mb": round(r.output_size_mb, 2),
                "compression_ratio": round(r.compression_ratio, 2),
                "error": r.error,
            }
            for r in results
        ],
    }
    (out / "report.json").write_text(json.dumps(report_json, indent=2))

    # report.md
    md_lines = [
        "# Data Pipeline Benchmark Report",
        "",
        f"**Generated**: {now}",
        f"**Source**: {source_path}",
        f"**Results**: {len(results)} runs",
        "",
        "## Format × Compression Comparison",
        "",
        "| Format | Compression | Mean Time (s) | Mean Size (MB) | Mean Ratio |",
        "|--------|-------------|---------------|----------------|------------|",
    ]
    for frow in fmt_rows:
        md_lines.append(
            f"| {frow['format']} | {frow['compression']} | {frow['mean_time_s']} | {frow['mean_size_mb']} | {frow['mean_compression_ratio']} |"
        )
    md_lines += [
        "",
        "## Backend Comparison",
        "",
        "| Backend | Runs | Mean Time (s) | Errors |",
        "|---------|------|---------------|--------|",
    ]
    for brow in be_rows:
        md_lines.append(
            f"| {brow['backend']} | {brow['n_runs']} | {brow['mean_time_s']} | {brow['n_errors']} |"
        )
    md_lines += [
        "",
        "## Per-Run Details",
        "",
        "| Format | Compression | Backend | Time (s) | Size (MB) | Ratio | Error |",
        "|--------|-------------|---------|----------|-----------|-------|-------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['format']} | {r['compression']} | {r['backend']} | {r['read_time_s']} | {r['output_size_mb']} | {r['compression_ratio']} | {r['error']} |"
        )

    (out / "report.md").write_text("\n".join(md_lines) + "\n")
    print(f"\nReports written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic data pipeline benchmark")
    parser.add_argument("--rows", type=int, default=1_000_000, help="Number of rows to generate (default: 1M)")
    parser.add_argument("--source", type=str, default=None, help="Use existing source file instead of generating")
    parser.add_argument("--formats", nargs="+", default=["parquet", "feather", "orc"],
                        choices=["csv", "parquet", "feather", "orc"], help="Storage formats to test")
    parser.add_argument("--compressions", nargs="+", default=["snappy", "zstd"],
                        choices=["none", "snappy", "zstd", "lz4", "gzip"], help="Compression codecs to test")
    parser.add_argument("--backends", nargs="+", default=["pandas", "pyarrow"],
                        choices=["pandas", "pyarrow", "polars", "dask", "vaex", "modin"], help="Backends to test")
    parser.add_argument("--output-root", type=str, default="./outputs/data-benchmarks",
                        help="Output root directory")
    parser.add_argument("--skip-generate", action="store_true", help="Skip data generation (use existing source)")
    args = parser.parse_args()

    # Resolve source
    source_path = args.source or _generate_synthetic_data(n_rows=args.rows)

    # Generate run ID
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_root, run_id)

    # Filter backends to available
    formats = [StorageFormat(f) for f in args.formats]
    compressions = [CompressionCodec(c) for c in args.compressions]
    backends = []
    for b in args.backends:
        be = ExecutionBackend(b)
        if is_backend_available(be):
            backends.append(be)
        else:
            print(f"WARNING: Backend '{b}' not available — skipping")

    if not backends:
        print("ERROR: No backends available. Install at least pandas.")
        sys.exit(1)

    print(f"Benchmark: {len(formats)} formats × {len(compressions)} compressions × {len(backends)} backends")
    print(f"Source: {source_path}")

    results = run_benchmarks(source_path, formats, compressions, backends)
    _write_reports(results, output_dir, source_path)

    # Summary
    ok = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]
    print(f"\nDone: {len(ok)} succeeded, {len(errors)} errors")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
