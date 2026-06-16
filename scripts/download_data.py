"""Dataset download script — 数据下载引导脚本。

当前项目使用 TAAC 数据集，数据文件需放置在 `./data/` 目录下。
TAAC 数据集为线下竞赛数据，请从对应渠道获取后手动放置。

已注册的数据集：
- taac2025_1M, taac2025_10M: TAAC 2025 数据适配器
- taac2026_data_sample, taac2026_second_round: TAAC 2026 数据适配器

使用方法：
1. 将数据文件放置到 ./data/ 目录下
2. 在 ExperimentConfig.data_config 中设置 root_dir 指向数据目录

其他公共数据集（MovieLens、Criteo、Taobao）尚未接通，
待数据集适配器实现后本脚本将提供自动下载功能。
"""

from __future__ import annotations


def main() -> None:
    print(__doc__)
    print("\n当前建议：")
    print("  1. 将 TAAC 数据放置到 ./data/ 目录")
    print("  2. 使用 run_single.py 或 Python API 直接运行实验")
    print()
    print("  示例:")
    print("    uv run python scripts/run_single.py --model itemcf --dataset taac2026_data_sample --seed 42")


if __name__ == "__main__":
    main()
