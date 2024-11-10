from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from transformers import EvalPrediction
import numpy as np

def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    """埋め込みと実際のラベルを用いたバッチ内でのkNNベースのF1スコアを計算"""
    embeddings = p.predictions
    labels = p.label_ids
    # 正解ラベルのインデックスを持つように変更
    ans_label = labels.argmax(axis=1)

    # 予測ラベル
    y_pred = []

    # 埋め込みに基づきk近傍法を適用
    for i in range(len(embeddings)):
      knn = KNeighborsClassifier(n_neighbors=3)
      # 予測対象を除く
      knn.fit(np.delete(embeddings, i ,axis=0), np.delete(ans_label, i ,axis=0))
      pred = knn.predict(embeddings[i].reshape(1, -1))
      y_pred.append(pred)

    # macro-F1スコア計算
    f1 = f1_score(ans_label, y_pred, average='macro')
    # F1スコアを返す
    return {"f1_score": f1}