import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.utils import ModelOutput
from transformers import BatchEncoding

def dice_coefficient(vec1, vec2):
    # 交差部分: 両方が1のインデックスの数
    intersection = sum(1 for a, b in zip(vec1, vec2) if a == 1 and b == 1)
    
    # ダイス係数の計算
    return 2 * intersection / (len(vec1) + len(vec2)) if (len(vec1) + len(vec2)) != 0 else 0


class SimCSEModel(nn.Module):
    """Sup_dscl SimCSEのモデル"""

    def __init__(
        self,
        base_model_name: str,
        mlp_only_train: bool = False,
        temperature: float = 0.05,
    ):
        """モデルの初期化"""
        super().__init__()

        # モデル名からエンコーダを初期化する
        self.encoder = AutoModel.from_pretrained(base_model_name)

        # MLP層の次元数
        self.hidden_size = self.encoder.config.hidden_size
        # MLP層の線形層
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        # MLP層の活性化関数
        self.activation = nn.Tanh()

        # MLP層による変換を訓練時にのみ適用するよう設定するフラグ
        self.mlp_only_train = mlp_only_train
        # 交差エントロピー損失の計算時に使用する温度
        self.temperature = temperature

    def encode_texts(self, tokenized_texts: BatchEncoding) -> Tensor:
        """エンコーダを用いて文をベクトルに変換"""
        # トークナイズされた文をエンコーダに入力する
        encoded_texts = self.encoder(**tokenized_texts)
        # モデルの最終層の出力（last_hidden_state）の
        # [CLS]トークン（0番目の位置のトークン）のベクトルを取り出す
        encoded_texts = encoded_texts.last_hidden_state[:, 0]

        # self.mlp_only_trainのフラグがTrueに設定されていて
        # かつ訓練時でない場合、MLP層の変換を適用せずにベクトルを返す
        if self.mlp_only_train and not self.training:
            return encoded_texts

        # MLP層によるベクトルの変換を行う
        encoded_texts = self.dense(encoded_texts)
        encoded_texts = self.activation(encoded_texts)

        return encoded_texts

    def forward(
        self,
        **inputs
    ) -> ModelOutput:
        """モデルの前向き計算（訓練と検証の両方に対応）"""

        # 訓練と検証の両方で使われる入力データを処理
        if 'tokenized_text' in inputs:
            # 検証用の処理
            tokenized_text = inputs["tokenized_text"]
            labels = inputs["labels"]
            encoded_text = self.encode_texts(tokenized_text)
            return ModelOutput(loss=torch.tensor(0.0), scores=encoded_text)

        # 訓練用の処理
        tokenized_texts_1 = inputs["tokenized_texts_1"]
        tokenized_texts_2 = inputs["tokenized_texts_2"]
        label = inputs["label_list"]
        labels = inputs["labels"]

        # 文ペアをベクトルに変換する
        encoded_texts_1 = self.encode_texts(tokenized_texts_1)
        encoded_texts_2 = self.encode_texts(tokenized_texts_2)

        # loss計算
        loss = 0
        for i in range(len(encoded_texts_1)):
            loss_i = 0
            # 分母の計算
            denominator = sum(
                torch.exp(F.cosine_similarity(encoded_texts_1[i].unsqueeze(0), encoded_texts_2[j].unsqueeze(0)) / self.temperature)
                for j in range(len(encoded_texts_1))
            )   
            # 交差エントロピー損失を計算
            for s in labels[i]:
                # 同じバッチ内で他のサンプルとの比較
                dice = dice_coefficient(label[i], label[s])
                # コサイン類似度
                sim_ij = F.cosine_similarity(encoded_texts_1[i].unsqueeze(0), encoded_texts_2[s].unsqueeze(0)) / self.temperature
                sim_ij = torch.exp(sim_ij)
                # ロスの計算
                loss_i += dice * torch.log(sim_ij / denominator)
                
            loss += -1 * loss_i / len(labels[i])

        return ModelOutput(loss=loss)