"""HyFormer 模型演示脚本。

展示：
1. HyFormer 模型的基本使用
2. 稀疏/密集参数分离的训练方式
3. Focal Loss 的使用
"""

import torch
from torch.optim import Adagrad, AdamW

from recsys import auto_discover_models, get_model
from recsys.core.base_model import Batch
from recsys.training.losses import sigmoid_focal_loss


def main() -> None:
    # 自动发现模型
    auto_discover_models()

    # 创建模型
    hyformer_cls = get_model("hyformer")
    model = hyformer_cls(
        config={"d_model": 64, "emb_dim": 64},
        schema_metadata={"num_users": 1000, "num_items": 5000},
    )

    print(f"Model: {model.model_name}")
    print(f"Family: {model.model_family}")
    print(f"Task type: {model.task_type}")
    print(f"Supports training: {model.supports_training}")

    # 创建模拟数据
    batch = Batch(
        data={
            "user_id": torch.randint(1, 1000, (32,)),
            "item_id": torch.randint(1, 5000, (32,)),
            "label": torch.randint(0, 2, (32,)).float(),
        }
    )

    # 前向传播
    output = model(batch)
    print(f"\nScores shape: {output.scores.shape}")
    print(f"Scores range: [{output.scores.min().item():.4f}, {output.scores.max().item():.4f}]")

    # 计算损失
    loss_dict = model.compute_loss(batch, output)
    print(f"\nLoss: {loss_dict['loss'].item():.4f}")

    # 展示稀疏/密集参数分离
    sparse_params = model.get_sparse_params()
    dense_params = model.get_dense_params()
    print(f"\nSparse params: {len(sparse_params)} tensors")
    print(f"Dense params: {len(dense_params)} tensors")

    # 展示双优化器训练
    sparse_optimizer = Adagrad(sparse_params, lr=0.05)
    dense_optimizer = AdamW(dense_params, lr=0.001)

    # 单步训练
    sparse_optimizer.zero_grad()
    dense_optimizer.zero_grad()
    loss_dict["loss"].backward()
    sparse_optimizer.step()
    dense_optimizer.step()
    print("\nSingle training step completed")

    # 展示 Focal Loss
    logits = output.scores
    targets = batch.data["label"]
    focal_loss = sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0)
    print(f"\nFocal Loss: {focal_loss.item():.4f}")


if __name__ == "__main__":
    main()
