"""
Author: ByronVon
Date: 2023-08-18 00:58:22
FilePath: /exercises/models/utc/train.py
Description: 
"""
import argparse

import torch
from losses import cross_entropy_loss
from model import DateConversionModelv2
from pipe import Trainer
from rich import print
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from utils import datetime_accuracy


def load_dataset(fpath):
    data = []
    with open(fpath, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            data.append(line)

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test


class DateDataset(Dataset):
    def __init__(self, data, tokenizer, max_length) -> None:
        # data的格式: [(input_str, output_str), ...,]
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        itn, tn, machine_dates = zip(*data)
        # 预先编码，批量tokenize
        self.input_data = tokenizer(
            list(itn + tn),
            padding="max_length",
            max_length=max_length,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt",
        )

        self.target_data = torch.tensor(
            [[int(digit) for digit in date] for date in machine_dates + machine_dates], dtype=torch.long
        )

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, index):
        # 从字典中提取每个项目
        input_item = {key: val[index] for key, val in self.input_data.items()}
        return input_item, self.target_data[index]


def main():
    # 参数，后续改为从argparser中读取
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=32, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use.")
    parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--step", type=int, default=200, help="Step size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--temp_dir", default="./utc_ce_small_att", help="Temporary directory path.")
    parser.add_argument(
        "--pretrained_model_name",
        default="prajjwal1/bert-small",
        help="Name of the pretrained model.",
    )
    parser.add_argument("--dataset", type=str, default="./datetime.tsv")

    args = parser.parse_args()

    print(f"*******Prepare Dataset*********")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)

    train, test = load_dataset(args.dataset)
    print(f"top3 samples: {test[:3]}")

    trainset = DateDataset(train, tokenizer, args.max_len)
    testset = DateDataset(test, tokenizer, args.max_len)
    print("top3 samples for inputs", testset[:3])

    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    print(f"*******Load Model*********")
    # '11 nov 2023'->YYYYMMDD，其中Y/M/D在为[0-9]
    model = DateConversionModelv2(args.pretrained_model_name, 10, 8, tokenizer.cls_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.step // 5, num_training_steps=len(train_dataloader) * args.epoch
    )
    # loss_function = masked_cross_entropy_loss  # 请根据你的实际需求选择或定制损失函数
    loss_function = cross_entropy_loss  # 加了teacher forcing不太需要用masked_cross_entropy
    metrics = datetime_accuracy

    # 初始化训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        metrics=metrics,
        epoch=args.epoch,
        step=args.step,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=args.device,
        checkpoint_dir=args.temp_dir,
    )

    print("*******Training*********")
    trainer.train_epoch(checkpoint_path=args.temp_dir)


if __name__ == "__main__":
    main()
