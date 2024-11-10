from datasets import load_dataset

from transformers.trainer_utils import set_seed

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

valid_dataset = valid_dataset

# 訓練セットの形式と事例数・単一ラベル数を確認する
print(train_dataset)
print(valid_dataset)
print(test_dataset)

from utils.freq_label import freq_labeling
from utils.label_count import label_count

label_count_list = label_count(valid_dataset)

valid_dataset = valid_dataset.map(lambda example: freq_labeling(example, label_count_list))

print(valid_dataset)