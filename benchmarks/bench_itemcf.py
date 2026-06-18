"""ItemCF 性能对比基准：baseline vs optimized vs distributed（Part D2）。

复用项目已有的 run_benchmark() 基础设施进行矩阵式对比，
生成 summary.csv + leaderboard.csv + performance_comparison.md。

用法:
    uv run python benchmarks/bench_itemcf.py --scale 1m
    uv run python benchmarks/bench_itemcf.py --scale 10m
    uv run python benchmarks/bench_itemcf.py --scale 100k  # 快速测试
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# 添加 src 到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recsys.pipeline.benchmark import BenchmarkConfig, ResumeMode, run_benchmark

# 数据规模映射
SCALE_MAP = {
    "100k": {"num_users": 5000, "num_items": 2000, "num_interactions": 100_000},
    "1m": {"num_users": 10000, "num_items": 5000, "num_interactions": 1_000_000},
    "10m": {"num_users": 50000, "num_items": 20000, "num_interactions": 10_000_000},
}


def build_itemcf_benchmark_config(scale: str) -> BenchmarkConfig:
    """构建 ItemCF 性能对比矩阵配置。

    对比维度:
        1. baseline: cosine + weighted_sum + numpy backend
        2. optimized: cosine + weighted_sum + normalize(optional)
        3. iuf: IUF weighted similarity

    每组通过 model_config 参数注入不同优化组合。
    """
    scale_data = SCALE_MAP.get(scale)
    if scale_data is None:
        valid = ", ".join(SCALE_MAP.keys())
        raise ValueError(f"无效的 scale '{scale}'，可选: {valid}")

    return BenchmarkConfig(
        benchmark_name=f"itemcf_perf_{scale}",
        models=["itemcf"],  # 通过 model_config 切换优化参数
        datasets=["synthetic"],  # 合成数据集
        seeds=[42],
        experiment_preset="itemcf_perf_comparison",
        resume_mode=ResumeMode.FORCE,
        max_concurrent_runs=1,  # 性能测试串行，避免干扰
        output_root="./outputs/benchmarks",
    )


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """预设多组 model_config 对比方案。

    Returns
    -------
    dict
        配置名 -> model_config 参数字典。
    """
    return {
        "itemcf_baseline": {
            "params": {
                "similarity": "cosine",
                "top_k_neighbors": 50,
                "recommend_k": 10,
                "prediction_method": "weighted_sum",
                "compute_backend": "numpy",
                "normalize": False,
            },
        },
        "itemcf_optimized": {
            "params": {
                "similarity": "cosine",
                "top_k_neighbors": 50,
                "recommend_k": 10,
                "prediction_method": "weighted_sum",
                "compute_backend": "numpy",
                "normalize": True,  # max 归一化（非论文方法，实验性）
            },
        },
        "itemcf_iuf": {
            "params": {
                "similarity": "iuf",
                "top_k_neighbors": 50,
                "recommend_k": 10,
                "prediction_method": "weighted_sum",
                "compute_backend": "numpy",
                "normalize": False,
            },
        },
    }


def get_data_configs(scale: str) -> Dict[str, Any]:
    """获取不同规模的数据集配置。"""
    scale_data = SCALE_MAP.get(scale, SCALE_MAP["1m"])
    return {
        "num_users": scale_data["num_users"],
        "num_items": scale_data["num_items"],
        "num_interactions": scale_data["num_interactions"],
        "seed": 42,
        "popularity_power": 0.75,
    }


def main():
    """主入口。"""
    parser = argparse.ArgumentParser(
        description="ItemCF 性能对比基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    uv run python benchmarks/bench_itemcf.py --scale 100k   # 快速测试 (10万交互)
    uv run python benchmarks/bench_itemcf.py --scale 1m     # 标准测试 (100万交互)
    uv run python benchmarks/bench_itemcf.py --scale 10m    # 大规模测试 (1000万交互)
        """,
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="100k",
        choices=list(SCALE_MAP.keys()),
        help="数据规模 (默认: 100k)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/benchmarks",
        help="输出目录 (默认: ./outputs/benchmarks)",
    )
    args = parser.parse_args()

    scale = args.scale
    scale_data = SCALE_MAP[scale]

    print("ItemCF 性能对比基准测试")
    print(f"  数据规模: {scale} (用户={scale_data['num_users']}, 物品={scale_data['num_items']}, 交互={scale_data['num_interactions']})")
    print(f"  输出目录: {args.output}")
    print()

    # 构建 BenchmarkConfig
    bench_cfg = BenchmarkConfig(
        benchmark_name=f"itemcf_perf_{scale}",
        models=["itemcf"],
        datasets=["synthetic"],
        seeds=[42],
        experiment_preset="itemcf_perf_comparison",
        resume_mode=ResumeMode.FORCE,
        max_concurrent_runs=1,
        output_root=args.output,
    )

    # 运行 benchmark
    print("开始运行 benchmark...")
    result = run_benchmark(bench_cfg)

    # 输出结果
    print("\nBenchmark 完成!")
    print(f"  状态: {result.status}")
    print(f"  总运行数: {len(result.runs)}")
    succeeded = sum(1 for r in result.runs if r.succeeded)
    failed = sum(1 for r in result.runs if r.status.value == "failed")
    print(f"  成功: {succeeded}, 失败: {failed}")
    print(f"  汇总文件: {result.summary_path}")
    print(f"  排行榜文件: {result.leaderboard_path}")

    if result.manifest_path:
        print(f"  Manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
