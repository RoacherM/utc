"""
Author: ByronVon
Date: 2023-08-18 17:31:33
FilePath: /exercises/models/utc/test.py
Description:
"""
import torch
from model import DateConversionModelv2, DateConversionModelv3
from pipe import Predictor
from rich import print
from transformers import AutoTokenizer

ordinal_to_cardinal = {
    "first": "one",
    "second": "two",
    "third": "three",
    "fourth": "four",
    "fifth": "five",
    "sixth": "six",
    "seventh": "seven",
    "eighth": "eight",
    "ninth": "nine",
    "tenth": "ten",
    "eleventh": "eleven",
    "twelfth": "twelve",
    "thirteenth": "thirteen",
    "fourteenth": "fourteen",
    "fifteenth": "fifteen",
    "sixteenth": "sixteen",
    "seventeenth": "seventeen",
    "eighteenth": "eighteen",
    "nineteenth": "nineteen",
    "twentieth": "twenty",
}


def ordinal_to_cardinal_conversion(text):
    words = text.split()
    converted_words = [ordinal_to_cardinal.get(word, word) for word in words]
    return " ".join(converted_words)


if __name__ == "__main__":
    pretrained_model_name = "prajjwal1/bert-small"
    checkpoint_dir = "./utc_ce2_small_dropout"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = DateConversionModelv2(pretrained_model_name, 10, 8, tokenizer.cls_token_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    predictor = Predictor(model, tokenizer, device, checkpoint_dir)

    spoken_dates = [
        "January first two thousand twenty three",
        "first of January two thousand and twenty three",
        "January first",
        "first January",
        "February fourteenth two thousand and twenty three",
        "fourteenth of February two thousand and twenty three",
        "Valentine's Day twenty twenty three",
        "February fourteenth",
        "fourteenth February",
        "July fourth two thousand and twenty three",
        "fourth of July two thousand and twenty three",
        "July fourth",
        "fourth July",
        "October thirty first two thousand and twenty three",
        "thirty first of October two thousand and twenty three",
        "Halloween twenty twenty three",
        "October thirty first",
        "thirty first October",
        "December twenty fifth two thousand and twenty three",
        "twenty fifth of December two thousand and twenty three",
        "Christmas twenty twenty three",
        "December twenty fifth",
        "twenty fifth December",
        "on may five",
        "jan second",
        "feb",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]

    spoken_dates_ = [ordinal_to_cardinal_conversion(_) for _ in spoken_dates]
    predicts_dates = predictor.predict(spoken_dates_)
    print(list(zip(predicts_dates, spoken_dates_)))
