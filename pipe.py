import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from rich.progress import track
from transformers import AutoTokenizer
from utils import constrained_decoding


class EarlyStopping:
    def __init__(self, patience=5, checkpoint_path="best_model.pt", delta=0):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None
        self.delta = delta

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return False

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,  # 可以变为参数，从已有的optimizer/scheduler中选择答案
        scheduler,
        loss_function,
        metrics,  # 评估函数
        epoch,
        step,
        train_dataloader,
        test_dataloader,
        device,
        checkpoint_dir,
        early_stop_patience=None,
    ):  # 在这里添加一个 dataloader 参数
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.metrics = metrics
        self.epoch = epoch
        self.step = step
        self.train_dataloader = train_dataloader  # 将 dataloader 保存为一个实例变量
        self.test_dataloader = test_dataloader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        # 加入early_stop
        if early_stop_patience:
            self.early_stopper = EarlyStopping(
                patience=early_stop_patience, checkpoint_path=f"{self.checkpoint_dir}/best_model.pt"
            )
        else:
            self.early_stopper = None
        # 记录每个epoch的结果
        self.history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    def train_epoch(self, checkpoint_path=None):
        # self.logger.info(f"Training Epoch {epoch}...")
        current_step = 0
        # load lastest checkpoint
        if checkpoint_path:
            current_step = self.load_checkpoint(checkpoint_path)

        for e in range(self.epoch):
            self.model.train()
            total_loss = 0.0
            all_predictions = []
            all_targets = []
            for input_data, target_data in track(self.train_dataloader, description=f"Training {e+1} epoch"):
                input_data = {k: v.to(self.device) for k, v in input_data.items()}
                target_data = target_data.to(self.device)

                # Start training logic
                self.optimizer.zero_grad()
                # Assuming your model returns predictions
                predictions = self.model(**input_data)
                loss = self.loss_function(predictions, target_data)
                total_loss += loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Convert tensor predictions and targets to numpy and append to list
                all_predictions.append(predictions.argmax(dim=2).cpu().detach().numpy())
                all_targets.append(target_data.cpu().numpy())

                # Save checkpoint
                current_step += 1
                if current_step % self.step == 0:
                    self.save_checkpoint(current_step)

            avg_loss = total_loss / len(self.train_dataloader)
            print(f"Train loss: {avg_loss}")

            # Calculate accuracy after each epoch
            all_predictions = np.concatenate(all_predictions)
            all_targets = np.concatenate(all_targets)
            acc_results = self.metrics(all_predictions, all_targets)
            print(acc_results)

            # add to history
            self.history["train_loss"].append(avg_loss.item())
            self.history["train_acc"].append(acc_results)

            self.eval_model()
            self.save_history()  # 保存训练记录

            # early stopping after each epoch
            if self.early_stopper:
                should_stop = self.early_stopper.step(acc_results.get("Overall Accuracy", 0.0), self.model)
                if should_stop:
                    print("Early stopping triggered!")
                    return
            # self.logger.info(f"Epoch {epoch} completed.")

    def eval_model(self):
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for input_data, target_data in track(self.test_dataloader, description=f"Evaluating..."):
                input_data = {k: v.to(self.device) for k, v in input_data.items()}
                target_data = target_data.to(self.device)

                predictions = self.model(**input_data)
                loss = self.loss_function(predictions, target_data)
                total_loss += loss

                # Convert tensor predictions and targets to numpy and append to list
                all_predictions.append(predictions.argmax(dim=2).cpu().detach().numpy())
                all_targets.append(target_data.cpu().numpy())
        avg_loss = total_loss / len(self.test_dataloader)
        print(f"Eval loss: {avg_loss}")

        # Calculate accuracy after each epoch
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        acc_results = self.metrics(all_predictions, all_targets)
        print(acc_results)

        # add to history
        self.history["test_loss"].append(avg_loss.item())  #
        self.history["test_acc"].append(acc_results)

    def save_checkpoint(self, step, keep_lastest_n=5):
        # 1. 保存当前的检查点
        current_checkpoint = f"{self.checkpoint_dir}/checkpoint_step_{step}.pt"
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "step": step,
        }
        torch.save(checkpoint, current_checkpoint)

        # 2. 列出检查点目录中的所有检查点
        all_checkpoints = [
            f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_step_") and f.endswith(".pt")
        ]
        all_checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

        # 3. 如果检查点数量超过N，删除多余的检查点
        if len(all_checkpoints) > keep_lastest_n:
            for cp in all_checkpoints[:-keep_lastest_n]:  # only keep the last N
                os.remove(os.path.join(self.checkpoint_dir, cp))

    def load_checkpoint(self, directory):
        # 获取所有的checkpoint文件名
        checkpoints = [f for f in os.listdir(directory) if f.startswith("checkpoint_step_") and f.endswith(".pt")]

        # 如果没有找到checkpoint，则返回None或抛出异常
        if not checkpoints:
            print("No checkpoints found! Training from start.")
            return 0
        # 根据step对文件名进行排序
        checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

        # 选择最新的checkpoint文件
        step_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(directory, step_checkpoint)

        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        current_step = checkpoint["step"]

        print(f"Loaded checkpoint from {checkpoint_path}")
        return current_step

    def save_history(self):
        with open(f"{self.checkpoint_dir}/metrics.jsonl", "a") as jsonfile:
            lastest_data = {
                "train_loss": self.history["train_loss"][-1],
                "train_acc": self.history["train_acc"][-1],
                "test_loss": self.history["test_loss"][-1],
                "test_acc": self.history["test_acc"][-1],
            }
            jsonfile.write(json.dumps(lastest_data) + "\n")


class Predictor:
    def __init__(self, model, tokenizer, device, checkpoint_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self._load_last_checkpoint()

    def _load_last_checkpoint(self):
        # 获取所有的checkpoint文件名
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_step_") and f.endswith(".pt")
        ]

        # 如果没有找到checkpoint，则返回None或抛出异常
        if not checkpoints:
            raise Exception("No checkpoints found!")

        # 根据step对文件名进行排序
        checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

        # 选择最新的checkpoint文件
        last_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(self.checkpoint_dir, last_checkpoint)

        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded checkpoint from {checkpoint_path}")

    def predict(self, text):
        # 转换文本到模型所需要的输入格式
        # 这里假设你有一个函数叫做 `text_to_input_data` 来完成这个转换
        if isinstance(text, str):
            text = [text]

        input_data = self.tokenizer(
            text,
            padding="max_length",
            max_length=16,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt",
        )

        input_data = {k: v.to(self.device) for k, v in input_data.items()}

        with torch.no_grad():
            logits = self.model(**input_data)
            predictions = logits.argmax(dim=2).cpu().detach().numpy()

        # 将模型输出转换为 'YYYYMMDD' 格式
        formatted_date = constrained_decoding(predictions)

        return formatted_date
