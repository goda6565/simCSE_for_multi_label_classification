from datasets import load_dataset

from transformers.trainer_utils import set_seed
from transformers import AutoTokenizer

import lightgbm as lgb

# 乱数のシードを設定する
set_seed(42)

# データ読み込み
train_dataset = load_dataset(
    "Harutiin/eurlex-for-bert", split="train"
)
valid_dataset = load_dataset(
    "Harutiin/eurlex-for-bert", split="validation"
)
test_dataset = load_dataset(
    "Harutiin/eurlex-for-bert", split="test"
)

# 訓練セットの形式と事例数・単一ラベル数を確認する
print(train_dataset)
print(valid_dataset[0])
print(test_dataset)

base_model_name = "cl-nagoya/unsup-simcse-ja-base"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)