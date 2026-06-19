"""Dataset download script — 通过 HuggingFace datasets 库下载 TAAC 数据集。

Usage:
    uv run python scripts/download_data.py --dataset taac2026_data_sample
    uv run python scripts/download_data.py --dataset taac2025_1M --cache-dir ./data
    uv run python scripts/download_data.py --dataset taac2026_second_round
"""

from __future__ import annotations

import argparse
import sys

from datasets import load_dataset  # noqa: E402

# 注册名到 HuggingFace repo_id 的映射
_REPO_MAP: dict[str, str] = {
    "taac2025_1M": "TAAC2025/TencentGR-1M",
    "taac2025_10M": "TAAC2025/TencentGR-10M",
    "taac2026_data_sample": "TAAC2026/data_sample_1000",
    "taac2026_second_round": "TAAC2026/second_round_sample_1000",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通过 HuggingFace 下载 TAAC 数据集到本地缓存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        choices=list(_REPO_MAP.keys()),
        help="数据集注册名",
    )
    parser.add_argument(
        "--cache-dir",
        default="./data",
        help="缓存目录 (默认: ./data)",
    )
    args = parser.parse_args()

    repo_id = _REPO_MAP[args.dataset]
    print(f"Downloading {repo_id} ...")
    try:
        ds = load_dataset(
            repo_id, "default", split="train", cache_dir=args.cache_dir,
        )
    except ImportError:
        print(
            "Error: `datasets` 未安装。请运行: uv pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    n_cols = len(ds.features) if hasattr(ds, "features") else len(ds.column_names)
    print(f"  Rows: {len(ds):,}, Columns: {n_cols}")
    print(f"  Cache location: {args.cache_dir}")


if __name__ == "__main__":
    main()
