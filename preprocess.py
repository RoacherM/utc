"""
Author: ByronVon
Date: 2023-08-18 23:16:09
FilePath: /exercises/models/utc/demo.py
Description: 
"""
import re
import datetime
import random
from collections import defaultdict

from babel.dates import format_date
from tqdm import tqdm
from num2words import num2words

random.seed(12345)

LANG = "en"
LOCALES = "en_US"

FORMATS_TO_REGEX = {
    "[d] MMM [YYY]": re.compile(r"\[(?P<day>\d{1,2})\] \w+ \[(?P<year>\d{4})\]"),
    "[dd] MMM [YYY]": re.compile(r"\[(?P<day>\d{2})\] \w+ \[(?P<year>\d{4})\]"),
    "[d] MMMM [YYY]": re.compile(r"\[(?P<day>\d{1,2})\] \w+ \[(?P<year>\d{4})\]"),
    "[dd] MMMM [YYY]": re.compile(r"\[(?P<day>\d{2})\] \w+ \[(?P<year>\d{4})\]"),
    "MMM [d] [YYY]": re.compile(r"\w+ \[(?P<day>\d{1,2})\] \[(?P<year>\d{4})\]"),
    "MMM [dd] [YYY]": re.compile(r"\w+ \[(?P<day>\d{2})\] \[(?P<year>\d{4})\]"),
    "MMMM [d] [YYY]": re.compile(r"\w+ \[(?P<day>\d{1,2})\] \[(?P<year>\d{4})\]"),
    "MMMM [dd] [YYY]": re.compile(r"\w+ \[(?P<day>\d{2})\] \[(?P<year>\d{4})\]"),
    "MMM [YYY]": re.compile(r"\w+ \[(?P<year>\d{4})\]"),
    "MMMM [YYY]": re.compile(r"\w+ \[(?P<year>\d{4})\]"),
    "[d] MMM": re.compile(r"\[(?P<day>\d{1,2})\] \w+"),
    "[d] MMMM": re.compile(r"\[(?P<day>\d{1,2})\] \w+"),
    "[dd] MMM": re.compile(r"\[(?P<day>\d{2})\] \w+"),
    "[dd] MMMM": re.compile(r"\[(?P<day>\d{2})\] \w+"),
    "[YYY]": re.compile(r"\[(?P<year>\d{4})\]"),
    "MMMM": re.compile(r"\w+"),
    "MMM": re.compile(r"\w+"),
    "[dd]": re.compile(r"\[(?P<day>\d{2})\]"),
    "[d]": re.compile(r"\[(?P<day>\d{1,2})\]"),
}


def dates_from_year(year):
    """
    Generates all dates for a given year.
    """
    d = datetime.date(year, 1, 1)
    while d.year == year:
        yield d
        d += datetime.timedelta(days=1)


def generate_dates(dt):
    """
    Convert the provided date to different formats.
    """
    selected_format = random.choice(list(FORMATS_TO_REGEX.keys()))
    human_readable = format_date(dt, format=selected_format, locale=LOCALES).lower()
    itn = human_readable

    regex_pattern = FORMATS_TO_REGEX.get(selected_format, None)

    tn = itn
    if regex_pattern:
        matched_dict = regex_pattern.search(itn)
        if matched_dict:
            matched_dict = matched_dict.groupdict()
            # 可以修改这部分的逻辑，补充一些口语化的说法
            # 只保留cardial的说法，输入时做下词行的变换
            if "day" in matched_dict:
                word_day = num2words(int(matched_dict["day"]), lang=LANG)
                tn = tn.replace(f'[{matched_dict["day"]}]', word_day)
            if "year" in matched_dict:
                word_year = num2words(int(matched_dict["year"]), lang=LANG)
                tn = tn.replace(f'[{matched_dict["year"]}]', word_year)

    machine_readable_year = dt.strftime("%Y")
    machine_readable_month = dt.strftime("%m")
    machine_readable_day = dt.strftime("%d")

    if "YYY" not in selected_format:
        machine_readable_year = "0000"
    if "MMM" not in selected_format and "MMMM" not in selected_format:
        machine_readable_month = "00"
    if "d" not in selected_format and "dd" not in selected_format:
        machine_readable_day = "00"

    machine_readable = machine_readable_year + machine_readable_month + machine_readable_day

    itn = itn.replace("[", "").replace("]", "")

    return itn, tn, machine_readable


def analyze_dates(data):
    month_freq = defaultdict(int)
    day_freq = defaultdict(int)

    for _, _, machine_readable in data:
        # 格式应为 YYYY-MM-DD

        year = machine_readable[:4]
        month = machine_readable[4:6]
        day = machine_readable[6:]

        month_freq[month] += 1
        day_freq[day] += 1

    return month_freq, day_freq


def main():
    start_year = 1000
    end_year = 5000
    sampel_days_per_year = 50

    data = []

    for year in tqdm(range(start_year, end_year)):
        all_dates = list(dates_from_year(year))
        sampled_dates = random.sample(all_dates, min(sampel_days_per_year, len(all_dates)))

        for dt in sampled_dates:
            itn, tn, machine_readable = generate_dates(dt)
            data.append((itn, tn, machine_readable))

    month_freq, day_freq = analyze_dates(data)

    # 打印统计结果
    print("Month Frequencies:")
    for month, freq in sorted(month_freq.items(), key=lambda x: int(x[0])):
        print(f"Month {month}: {freq} times")

    print("\nDay Frequencies:")
    for day, freq in sorted(day_freq.items(), key=lambda x: int(x[0])):
        print(f"Day {day.zfill(2)}: {freq} times")

    # 保存文件
    with open("datetime2.tsv", "w") as f:
        for d in data:
            f.writelines(f"{d[0]}\t{d[1]}\t{d[2]}\n")


if __name__ == "__main__":
    # test_strings = {
    #     "[d] MMM [YYY]": "[1] Jan [2022]",
    #     "[dd] MMM [YYY]": "[01] Jan [2022]",
    #     "[d] MMMM [YYY]": "[1] January [2022]",
    #     "[dd] MMMM [YYY]": "[01] January [2022]",
    #     "[dd]": "[01]",
    #     "[d]": "[1]",
    # }

    # for format_, test_string in test_strings.items():
    #     match = FORMATS_TO_REGEX[format_].match(test_string)
    #     if match:
    #         print(f"Matching {test_string} using {format_}:")
    #         print(match.groupdict())
    #     else:
    #         print(f"Failed to match {test_string} using {format_}")

    main()
