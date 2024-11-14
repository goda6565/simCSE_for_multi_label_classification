from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import EvalPrediction
import numpy as np

def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    """kNN"""
    embeddings = p.predictions
    labels = p.label_ids

    # One-hotラベルを整数ラベルに変換
    integer_labels = np.argmax(labels, axis=1)
    
    # 埋め込みに基づきk近傍法を適用
    X_train, X_test, y_train, y_test = train_test_split(embeddings, integer_labels, test_size=0.2, random_state=42)
    kNN = KNeighborsClassifier(n_neighbors=3)
    kNN.fit(X_train, y_train)
    y_pred = kNN.predict(X_test)
    
    # macro-F1スコア計算
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # F1スコアを返す
    return {"f1_score": f1}