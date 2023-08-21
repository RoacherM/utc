"""
Author: ByronVon
Date: 2023-08-17 17:43:35
FilePath: /exercises/models/seq2seq/utils.py
Description: 
"""
import json

import matplotlib.pyplot as plt
import numpy as np


def constrained_decoding(predictions):
    """
    predictions: shape (batch_size, 8)
    """

    decoded_predictions = predictions

    # 月份
    month_predictions = decoded_predictions[:, 4:6]
    months = month_predictions[:, 0] * 10 + month_predictions[:, 1]
    # months[months < 1] = 1
    months[months > 12] = 12

    # 日期
    day_predictions = decoded_predictions[:, 6:8]
    days = day_predictions[:, 0] * 10 + day_predictions[:, 1]
    # days[days < 1] = 1
    days[days > 31] = 31

    # 更新解码的预测
    decoded_predictions[:, 4:6] = np.column_stack([(months // 10), (months % 10)])
    decoded_predictions[:, 6:8] = np.column_stack([(days // 10), (days % 10)])

    return decoded_predictions


def datetime_accuracy(predictions, targets):
    # 确保 predictions 和 targets 有相同的 shape
    assert predictions.shape == targets.shape

    correct_predictions = (predictions == targets).astype(int)
    # 整体准确率
    total_correct = correct_predictions.sum()
    total_accuracy = total_correct / predictions.size

    # 年、月、日的准确率
    year_correct = correct_predictions[:, :4].sum(axis=1)
    year_accuracy = np.mean(year_correct == 4)  # 全部正确为1的个数

    month_correct = correct_predictions[:, 4:6].sum(axis=1)
    month_accuracy = np.mean(month_correct == 2)  # 全部为1的个数

    day_correct = correct_predictions[:, 6:].sum(axis=1)
    day_accuracy = np.mean(day_correct == 2)  # 全部为1的个数

    return {
        "Overall Accuracy": total_accuracy,
        "Year Accuracy": year_accuracy,
        "Month Accuracy": month_accuracy,
        "Day Accuracy": day_accuracy,
    }


def visulize_train_history(fpath):
    with open(fpath, "r") as f:
        lines = f.readlines()
    data = [json.loads(l) for l in lines]
    # 损失图
    epochs = list(range(1, len(data) + 1))
    train_losses = [d["train_loss"] for d in data]
    test_losses = [d["test_loss"] for d in data]

    # 创建准确性和损失的双y轴图
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制损失
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epochs, train_losses, "o-", color="tab:blue", label="Train Loss")
    ax1.plot(epochs, test_losses, "s-", color="tab:cyan", label="Test Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    # 创建第二个y轴来表示准确性
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:red")

    # 为了示例，这里我们只绘制"Overall Accuracy"，但可以通过循环绘制所有准确性指标
    train_overall_accs = [d["train_acc"]["Overall Accuracy"] for d in data]
    test_overall_accs = [d["test_acc"]["Overall Accuracy"] for d in data]

    ax2.plot(epochs, train_overall_accs, "o-", color="tab:red", label="Train Overall Accuracy")
    ax2.plot(epochs, test_overall_accs, "s-", color="tab:orange", label="Test Overall Accuracy")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    plt.title("Loss and Accuracy vs Epochs")
    plt.grid(True)
    # plt.show()
    plt.savefig(f"{fpath.replace('jsonl', 'png')}")


if __name__ == "__main__":
    # # 示例
    # predictions = np.array([[2, 0, 2, 3, 0, 8, 1, 5], [2, 0, 2, 3, 0, 8, 1, 6]])
    # targets = np.array([[2, 0, 2, 3, 0, 6, 1, 6], [2, 0, 2, 3, 0, 8, 1, 5]])

    # print(accuracy(predictions, targets))

    visulize_train_history("./utc_ce3_small_dropout/metrics.jsonl")
