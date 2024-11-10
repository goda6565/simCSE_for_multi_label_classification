from transformers import AutoTokenizer

# モデル読み込み
base_model_name = "cl-nagoya/unsup-simcse-ja-base"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)