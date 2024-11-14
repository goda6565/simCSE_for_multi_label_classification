from transformers import EvalPrediction
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    """最も埋め込みの距離が近いテキストとのJaccard 係数の平均をとる"""
    embeddings = p.predictions
    labels = p.label_ids
    jaccard_means = []
    
    # 埋め込みのコサイン類似度をとる
    similarity_matrix = cosine_similarity(embeddings, embeddings)
    
    for i in range(len(embeddings)):
        # ターゲットラベルのワンホットインデックスを集合として取得
        target_label = set(np.nonzero(labels[i])[0])
        
        # 自分以外で最も類似度が高いインデックスを取得
        nearest_idx = np.argsort(similarity_matrix[i])[-2]
        # 類似度の高いラベルもワンホットインデックスに変換
        nearest_label = set(np.nonzero(labels[nearest_idx])[0])
        
        # Jaccard係数の計算
        intersection = target_label & nearest_label
        union = target_label | nearest_label
        jaccard = len(intersection) / len(union) if union else 0  # Unionが空なら0とする
        
        jaccard_means.append(jaccard)
        
    return {"jaccard_means": np.mean(jaccard_means)}