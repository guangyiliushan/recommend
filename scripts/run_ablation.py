"""Ablation study runner — 消融实验引导脚本。

消融实验需要系统性地控制变量并比较多个实验组，
当前建议使用 run_benchmark() CLI 或 Python API 实现。

简易消融方式：
1. 用 run_benchmark.py 传入多个配置变体
2. 比较不同模型/参数在 leaderboard.csv 上的表现
3. 结合 trend.csv 分析逐 seed 稳定性

完整的消融实验框架将在后续版本中实现，支持：
  - 组件级消融 (e.g. embedding_dim, hidden_dims, dropout)
  - 自动对比不同配置变体
  - 可视化消融结果

使用方法：
  uv run python scripts/run_ablation.py --help  (本脚本待实现)
"""

from __future__ import annotations


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
