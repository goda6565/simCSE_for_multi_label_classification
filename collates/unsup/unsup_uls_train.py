import torch
from torch import Tensor
from transformers import BatchEncoding
from config.princeton import tokenizer

def unsup_uls_train_collate_fn(
    examples: list[dict],
) -> dict[str, BatchEncoding | Tensor]:
    """Unique Label Samplingでマルチラベル訓練セットのミニバッチを作成"""
    # 全てのラベルが異なるサンプルのみを選択
    unique_label_examples = []
    seen_labels = []

    for example in examples:
        label = example["labels"]  # 各サンプルのラベル
        if label not in seen_labels:
            unique_label_examples.append(example)
            seen_labels.append(label)

        if len(seen_labels) == 64: # バッチサイズ
            break

    # バッチに含まれる文にトークナイザを適用
    tokenized_texts = tokenizer(
        [example["text"] for example in unique_label_examples],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )


    # 文と文の類似度行列における正例ペアの位置を示すTensorを作成
    labels = torch.arange(len(unique_label_examples))

    return {
        "tokenized_texts_1": tokenized_texts,
        "tokenized_texts_2": tokenized_texts,
        "labels": labels,
    }