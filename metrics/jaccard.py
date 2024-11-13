from transformers import EvalPrediction
import numpy as np

def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    """埋め込みと実際のラベルを用いたバッチ内でのkNNベースのF1スコアを計算"""