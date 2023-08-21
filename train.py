"""
Author: ByronVon
Date: 2023-08-18 00:58:22
FilePath: /exercises/models/utc/train.py
Description: 
"""

import torch
from losses import cross_entropy_loss, masked_cross_entropy_loss
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
    max_len = 32
    batch_size = 128
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    epoch = 10
    step = 200
    lr = 2e-5
    temp_dir = "./utc_ce2_xlmr_small_dropout"
    pretrained_model_name = "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large"

    print(f"*******Prepare Dataset*********")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    train, test = load_dataset("./datetime2.tsv")
    print(f"top3 samples: {test[:3]}")

    trainset = DateDataset(train, tokenizer, max_len)
    testset = DateDataset(test, tokenizer, max_len)
    print("top3 samples for inputs", testset[:3])

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    print(f"*******Load Model*********")
    # '11 nov 2023'->YYYYMMDD，其中Y/M/D在为[0-9]
    model = DateConversionModelv2(pretrained_model_name, 10, 8, tokenizer.cls_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=step // 5, num_training_steps=len(train_dataloader) * epoch
    )
    # loss_function = masked_cross_entropy_loss  # 请根据你的实际需求选择或定制损失函数
    loss_function = cross_entropy_loss  # 加了teacher forcing后，感觉不太需要用masked_cross_entropy # update: 确实不太需要
    metrics = datetime_accuracy

    # 初始化训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        metrics=metrics,
        epoch=epoch,
        step=step,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        checkpoint_dir=temp_dir,
    )

    print("*******Training*********")
    trainer.train_epoch(checkpoint_path=temp_dir)


if __name__ == "__main__":
    main()
