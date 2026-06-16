"""HyFormer 模型训练脚本。

支持：
1. 配置文件
2. 命令行参数
3. 稀疏/密集参数分离训练
4. Focal Loss
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import torch
from torch.optim import Adagrad, AdamW
from torch.utils.data import DataLoader, TensorDataset

from recsys import auto_discover_models, get_model
from recsys.core.base_model import Batch
from recsys.training.losses import sigmoid_focal_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HyFormer model")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_users", type=int, default=10000)
    parser.add_argument("--num_items", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sparse_lr", type=float, default=0.05)
    parser.add_argument("--loss_type", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def create_dummy_dataloader(
    num_users: int,
    num_items: int,
    batch_size: int,
    num_batches: int = 100,
) -> DataLoader:
    """创建模拟数据 DataLoader（演示用）。"""
    user_ids = torch.randint(1, num_users, (batch_size * num_batches,))
    item_ids = torch.randint(1, num_items, (batch_size * num_batches,))
    labels = torch.randint(0, 2, (batch_size * num_batches,)).float()
    dataset = TensorDataset(user_ids, item_ids, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def compute_loss(
    model: torch.nn.Module,
    batch: Batch,
    output: Any,
    loss_type: str,
    focal_alpha: float,
    focal_gamma: float,
) -> torch.Tensor:
    """计算损失。"""
    if loss_type == "focal":
        logits = output.scores
        targets = batch.data.get("label", batch.data.get("labels"))
        return sigmoid_focal_loss(logits, targets, alpha=focal_alpha, gamma=focal_gamma)
    else:
        loss_dict = model.compute_loss(batch, output)
        return loss_dict["loss"]


def main() -> None:
    args = parse_args()

    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logging.info(f"Using device: {device}")

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 自动发现模型
    auto_discover_models()

    # 创建模型
    hyformer_cls = get_model("hyformer")
    model = hyformer_cls(
        config={
            "d_model": args.d_model,
            "emb_dim": args.emb_dim,
            "num_heads": args.num_heads,
            "num_blocks": args.num_blocks,
            "dropout": args.dropout,
        },
        schema_metadata={
            "num_users": args.num_users,
            "num_items": args.num_items,
        },
    ).to(device)

    logging.info(f"Model created: {model.model_name}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # 双优化器
    sparse_params = model.get_sparse_params()
    dense_params = model.get_dense_params()
    sparse_optimizer = Adagrad(sparse_params, lr=args.sparse_lr)
    dense_optimizer = AdamW(dense_params, lr=args.lr)

    logging.info(f"Sparse params: {len(sparse_params)} tensors (Adagrad, lr={args.sparse_lr})")
    logging.info(f"Dense params: {len(dense_params)} tensors (AdamW, lr={args.lr})")

    # 创建模拟数据
    train_loader = create_dummy_dataloader(
        num_users=args.num_users,
        num_items=args.num_items,
        batch_size=args.batch_size,
    )

    # 训练循环
    model.train()
    global_step = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for user_ids, item_ids, labels in train_loader:
            batch = Batch(
                data={
                    "user_id": user_ids.to(device),
                    "item_id": item_ids.to(device),
                    "label": labels.to(device),
                }
            )

            sparse_optimizer.zero_grad()
            dense_optimizer.zero_grad()

            output = model(batch)
            loss = compute_loss(
                model,
                batch,
                output,
                args.loss_type,
                args.focal_alpha,
                args.focal_gamma,
            )

            loss.backward()
            sparse_optimizer.step()
            dense_optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        avg_loss = epoch_loss / num_batches
        logging.info(f"Epoch {epoch + 1}/{args.num_epochs} - Loss: {avg_loss:.4f}")

    logging.info(f"Training complete! Total steps: {global_step}")


if __name__ == "__main__":
    main()
