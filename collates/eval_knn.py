import torch
from torch import Tensor
from transformers import BatchEncoding
from config.princeton import tokenizer

def eval_collate_fn(
    examples: list[dict],
) -> dict[str, BatchEncoding | Tensor]:
    """SimCSEの検証・テストセットのミニバッチを作成"""
    # ミニバッチの文ペアに含まれる文（文1と文2）のそれぞれに
    # トークナイザを適用する
    tokenized_texts = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # データセットに付与されたラベル配列のTensorを作成する
    label = torch.tensor(
        [example["labels"] for example in examples]
    )

    return {
        "tokenized_text": tokenized_texts,
        "labels": label,
    }