from transformers import TrainingArguments

# 訓練のハイパーパラメータを設定する
training_args = TrainingArguments(
    output_dir="outputs/sup_pe",  # 結果の保存先フォルダ
    per_device_train_batch_size=32,  # 訓練時のバッチサイズ
    per_device_eval_batch_size=64,  # 評価時のバッチサイズ
    learning_rate=1e-5,  # 学習率
    num_train_epochs=1,  # 訓練エポック数（step数）
    eval_strategy="steps",  # 検証セットによる評価のタイミング
    eval_steps=120,  # 検証セットによる評価を行う訓練ステップ数の間隔
    logging_steps=120,  # ロギングを行う訓練ステップ数の間隔
    save_steps=120,  # チェックポイントを保存する訓練ステップ数の間隔
    save_total_limit=1,  # 保存するチェックポイントの最大数
    fp16=True,  # 自動混合精度演算の有効化
    load_best_model_at_end=True,  # 最良のモデルを訓練終了後に読み込むか
    metric_for_best_model="f1_score",  # 最良のモデルを決定する評価指標
    label_names=["labels"], # ラベルを指定マルチラベル
    remove_unused_columns=False,  # データセットの不要フィールドを削除するか
    report_to="wandb" 
)