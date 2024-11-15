import xgboost as xgb
from datasets import load_dataset
from transformers.trainer_utils import set_seed
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from classifications.emb import embed_texts


# 乱数のシードを設定する
set_seed(42)

# データ読み込み
train_dataset = load_dataset(
    "Harutiin/eurlex-for-bert", split="train"
)
test_dataset = load_dataset(
    "Harutiin/eurlex-for-bert", split="test"
)


# データの作成
# 訓練
X_train = embed_texts(train_dataset["text"])

y_train = train_dataset["labels"]

# テスト
X_test = embed_texts(test_dataset["text"])

y_test = test_dataset["labels"]

# `y_train` と `y_val` を NumPy 配列に変換
y_train = np.array(y_train)
y_test = np.array(y_test)


# xgboost
# モデルの訓練と予測のためのループ（進捗バーを追加）

model = xgb.XGBClassifier(
    tree_method="hist",
    device="cuda"
)

# モデルを訓練
model.fit(X_train, y_train)

# 予測を行う
pred = model.predict(X_test)

# 結果を表示
print(f"Macro Precision: {precision_score(y_test, pred, average="macro",zero_division=0):.5f}")
print(f"Macro Recall: {recall_score(y_test, pred, average="macro",zero_division=0):.5f}")
print(f"Macro F1: {f1_score(y_test, pred, average="macro",zero_division=0):.5f}")
print(f"Micro F1: {f1_score(y_test, pred, average="micro",zero_division=0):.5f}")