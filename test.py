from datasets import load_dataset

from transformers.trainer_utils import set_seed
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from utils.single_label import random_choice

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

valid_dataset = valid_dataset.map(random_choice)

# 訓練セットの形式と事例数・単一ラベル数を確認する
print(train_dataset)
print(valid_dataset)
print(test_dataset)
print(valid_dataset[2])