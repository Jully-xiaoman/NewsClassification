import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def read_txt_to_df(file_path):
    rows = []
    bad_lines = []

    with open(file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("_!_")
            if len(parts) == 5:
                rows.append(parts)
            else:
                bad_lines.append((lineno, line.strip(), len(parts)))

    if bad_lines:
        print(f"[Warning] {file_path} 中有 {len(bad_lines)} 行格式异常，已跳过。")
        print("前3条异常示例：")
        for item in bad_lines[:3]:
            print(item)

    df = pd.DataFrame(rows, columns=[
        "news_id", "category_code", "category_name", "title", "keywords"
    ])
    return df


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def build_label_map(train_df):
    label_names = sorted(train_df["category_name"].unique())
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def create_datasets_and_loaders(config):
    train_df = read_txt_to_df(config["train_path"])
    dev_df = read_txt_to_df(config["dev_path"])
    test_df = read_txt_to_df(config["test_path"])

    label2id, id2label = build_label_map(train_df)

    train_texts = train_df["title"].tolist()
    train_labels = train_df["category_name"].map(label2id).tolist()

    # 检查 dev 集是否出现训练集未见标签
    dev_label_series = dev_df["category_name"].map(label2id)
    if dev_label_series.isnull().any():
        unknown_dev_labels = dev_df.loc[dev_label_series.isnull(), "category_name"].unique()
        raise ValueError(f"dev 集出现训练集未见标签: {unknown_dev_labels}")

    # 检查 test 集是否出现训练集未见标签
    test_label_series = test_df["category_name"].map(label2id)
    if test_label_series.isnull().any():
        unknown_test_labels = test_df.loc[test_label_series.isnull(), "category_name"].unique()
        raise ValueError(f"test 集出现训练集未见标签: {unknown_test_labels}")

    # 检查通过后再正常生成文本和标签
    dev_texts = dev_df["title"].tolist()
    dev_labels = dev_label_series.astype(int).tolist()

    test_texts = test_df["title"].tolist()
    test_labels = test_label_series.astype(int).tolist()

    
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])

    train_dataset = NewsDataset(
        train_texts, train_labels, tokenizer, max_length=config["max_length"]
    )
    dev_dataset = NewsDataset(
        dev_texts, dev_labels, tokenizer, max_length=config["max_length"]
    )
    test_dataset = NewsDataset(
        test_texts, test_labels, tokenizer, max_length=config["max_length"]
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=config["batch_size"], shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    return train_dataloader, dev_dataloader, test_dataloader, label2id, id2label