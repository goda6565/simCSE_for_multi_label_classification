import torch
from torch import Tensor
from transformers import BatchEncoding
from config.princeton import tokenizer

def sup_train_collate_fn(
    examples: list[dict],
) -> dict[str, BatchEncoding | Tensor]:
    """訓練セットのミニバッチを作成"""
    texts = []
    same_label_texts = []
    start_labels = [0]
    end_labels = []
    labels = []

    for i, example in enumerate(examples):
        texts.append(example["text"])
        if i != 0:
          start_labels.append(end_labels[i-1]+1)
        # same_label_sentenceを収集
        same_labels = [text["same_label_text"] for text in examples if text["labels"] == example["labels"]]
        same_label_texts.extend(same_labels)
        end_labels.append(len(same_labels)+start_labels[i]-1)

    # ミニバッチに含まれる前提文と仮説文にトークナイザを適用する
    tokenized_texts1 = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    tokenized_texts2 = tokenizer(
        same_label_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    for i in range(len(texts)):
      labels.append(torch.arange(start_labels[i], end_labels[i] + 1))

    return {
        "tokenized_texts_1": tokenized_texts1,
        "tokenized_texts_2": tokenized_texts2,
        "labels": labels,
    }