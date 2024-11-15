import torch
from torch import Tensor
from transformers import BatchEncoding
from config.princeton import tokenizer

def sup_train_collate_fn(
    examples: list[dict],
) -> dict[str, BatchEncoding | Tensor]:
    """訓練セットのミニバッチを作成"""
    same_label_index = []
    for example in examples:
        index = []
        for i, pair in enumerate(examples):
            if example["labels"] == pair["labels"]:
                index.append(i)
        same_label_index.append(index)

    # ミニバッチに含まれる前提文と仮説文にトークナイザを適用する
    tokenized_texts1 = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    tokenized_texts2 = tokenizer(
        [example["same_label_text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    labels = []
    for i in range(len(examples)):
      labels.append(torch.tensor(same_label_index[i]))

    return {
        "tokenized_texts_1": tokenized_texts1,
        "tokenized_texts_2": tokenized_texts2,
        "label_list": [example["labels"] for example in examples],
        "labels": labels,
    }