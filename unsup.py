from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.trainer_utils import set_seed

from utils.freq_label import freq_labeling
from utils.label_count import label_count
from config.unsup_args import training_args
from config.princeton import base_model_name
from metrics.knn_f1 import compute_metrics
from collates.eval_knn import eval_collate_fn
from collates.unsup.unsup_train import unsup_train_collate_fn
from models.unsup import SimCSEModel

import wandb

wandb.init(project="SimCSE-for -multilabel", name="unsup-base")

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

label_count_list = label_count(valid_dataset)
valid_dataset = valid_dataset.map(lambda example: freq_labeling(example, label_count_list))

# 教師なしSimCSEのモデルを初期化する
unsup_model = SimCSEModel(base_model_name, mlp_only_train=True)

# 訓練設定

class SimCSETrainer(Trainer):
    """SimCSEの訓練に使用するTrainer"""

    def get_eval_dataloader(
        self, eval_dataset: Dataset | None = None
    ) -> DataLoader:
        """
        検証・テストセットのDataLoaderでeval_collate_fnを使うように
        Trainerのget_eval_dataloaderをオーバーライド
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        return DataLoader(
            eval_dataset,
            batch_size=64,
            collate_fn=eval_collate_fn,
            pin_memory=True,
        )

# Trainerを初期化する
trainer = SimCSETrainer(
    model=unsup_model,
    args=training_args,
    data_collator=unsup_train_collate_fn,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

# パラメータを連続にする
for param in unsup_model.parameters():
    param.data = param.data.contiguous()
print(type(unsup_model).__name__)

# 教師なしSimCSEの訓練を行う
trainer.train()