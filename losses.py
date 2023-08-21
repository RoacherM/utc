"""
Author: ByronVon
Date: 2023-08-17 17:43:16
FilePath: /exercises/models/seq2seq/losses.py
Description: 
"""
import torch
import torch.nn.functional as F

# `F.cross_entropy(predictions.view(-1, 10), targets.view(-1))` 是在计算多分类交叉熵损失，其中：
# 1. `predictions`: 这是一个Tensor，它表示模型为每个可能的类别生成的概率分数。对于你的场景，它的形状为 `(batch_size, 8, 10)`。其中 `8` 是日期表示的字符数（例如 `YYYYMMDD`），`10` 是每个位置上可能的类别数（即 `0` 到 `9` 的10个数字）。
#    `predictions.view(-1, 10)` 将这个Tensor重塑为 `(batch_size * 8, 10)`，这样每行代表一个位置的类别分数，总共有 `batch_size * 8` 行。
# 2. `targets`: 这是一个Tensor，表示每个位置的真实类别（数字）。它的形状为 `(batch_size, 8)`，其中每个值都在 `0` 到 `9` 之间。
#    `targets.view(-1)` 将这个Tensor重塑为一个一维向量，长度为 `batch_size * 8`，与重塑后的 `predictions` 对应。
# `F.cross_entropy` 会为每一个位置计算交叉熵损失，并对所有位置取平均以得到最终的损失。
# 这个损失函数用于度量模型的预测分数与真实类别之间的差异。交叉熵损失越低，模型的预测越接近真实类别。


def masked_cross_entropy(predictions, targets, masks):
    """Compute masked cross entropy."""
    losses = F.cross_entropy(predictions.view(-1, 10), targets.view(-1), reduction="none")
    losses = losses.view(predictions.size(0), -1)
    masked_losses = losses * masks
    return masked_losses


def masked_cross_entropy_loss(predictions, targets):
    """
    Custom loss function for date prediction.

    Args:
    - predictions: Tensor of shape (batch_size, target_len, 10).
                   Represents predicted class probabilities for each position.
    - targets: Tensor of shape (batch_size, target_len).
               Represents the true labels.

    Returns:
    - final_loss: Scalar tensor. The computed loss value.
    """

    # Convert softmax predictions to actual number predictions
    pred_numbers = predictions.argmax(dim=2)  # [batch_size, 8]

    # Create a default mask of ones
    masks = torch.ones_like(targets).float()

    # Compute combined month values from predictions
    combined_month_values = pred_numbers[:, 4] * 10 + pred_numbers[:, 5]

    # For month values: Valid months are [1, 12]
    is_valid_month = (combined_month_values >= 1) & (combined_month_values <= 12)
    masks[:, 4:6] = masks[:, 4:6] * is_valid_month.unsqueeze(1).float()

    # Compute combined day values from predictions
    combined_day_values = pred_numbers[:, 6] * 10 + pred_numbers[:, 7]

    # For day values: Valid days are [1, 31]
    is_valid_day = (combined_day_values >= 1) & (combined_day_values <= 31)
    masks[:, 6:8] = masks[:, 6:8] * is_valid_day.unsqueeze(1).float()

    # Compute masked cross entropy loss
    masked_losses = masked_cross_entropy(predictions, targets, masks)

    # Compute the final loss: this is the mean of the masked losses.
    final_loss = masked_losses.sum() / masks.sum()

    return final_loss


def cross_entropy_loss(predictions, targets):
    """
    Standard loss function for date prediction without masking.

    Args:
    - predictions: Tensor of shape (batch_size, 8, 10).
                   Represents predicted class probabilities for each position.
    - targets: Tensor of shape (batch_size, 8).
               Represents the true labels.

    Returns:
    - loss: Scalar tensor. The computed loss value.
    """
    loss = F.cross_entropy(predictions.view(-1, 10), targets.view(-1))
    return loss


if __name__ == "__main__":
    # 示例
    predictions = torch.randn(32, 8, 10)  # batch of size 32
    targets = torch.randint(0, 10, (32, 8))
    print(predictions, targets)

    loss = masked_cross_entropy_loss(predictions, targets)
    print(loss)
