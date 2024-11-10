import torch
from torch import Tensor
from transformers import BatchEncoding
from config.princeton import tokenizer

def unsup_train_collate_fn(
    examples: list[dict],
) -> dict[str, BatchEncoding | Tensor]:
    """マルチラベル訓練セットのミニバッチを作成"""
    # ミニバッチに含まれる文にトークナイザを適用する
    tokenized_texts = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    # 文と文の類似度行列における正例ペアの位置を示すTensorを作成する
    # 行列のi行目の事例（文）に対してi列目の事例（文）との組が正例ペアとなる
    labels = torch.arange(len(examples))

    return {
        "tokenized_texts_1": tokenized_texts,
        "tokenized_texts_2": tokenized_texts,
        "labels": labels,
    }