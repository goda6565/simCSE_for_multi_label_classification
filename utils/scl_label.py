import random

def drop_unique_label(example, unique_label_dict):
    label = example["labels"]
    if unique_label_dict[str(label)] == 1:
        return False
    else:
        return True  # ユニークなラベルのサンプルは削除
    
def get_same_label(examples):
    same_label_dict = {}
    for example in examples:
        label = example["labels"]
        if label not in list(same_label_dict.keys()):
            same_label_dict[str(label)] = []
        same_label_dict[str(label)].append(example["text"])
    return same_label_dict

def set_same_label_text(example, same_label_dict):
    label = example["labels"]
    example["same_label_text"] = random.choice(same_label_dict[str(label)])
    return example
    

def create_same_label_datasets(examples):
    """各データにランダムで同じラベルを持つテキストを付与"""
    # ラベル一覧の取得
    unique_label_dict = {}
    for example in examples:
        label = example["labels"]
        unique_label_dict[str(label)] = unique_label_dict.get(str(label), 0) + 1
        

    # ラベルがユニークなものを削除
    filtered_examples = examples.filter(lambda example: drop_unique_label(example, unique_label_dict))
    
    same_label_dict = get_same_label(filtered_examples)
    
    annoteted_sample = filtered_examples.map(lambda example: set_same_label_text(example, same_label_dict))
    
    return annoteted_sample