import numpy as np

def freq_labeling(example, sorted_label_key):
    
    sorted_label_key
    
    label_indices = np.nonzero(example["labels"])[0]
    
    most_freq_label = 0 # ラベルのインデックス
    most_freq_label_rank = 100 # ラベルの順位
    
    for label in label_indices:
        if most_freq_label_rank >= sorted_label_key.index(label): # 順位
            most_freq_label = label
            most_freq_label_rank = sorted_label_key.index(label)
    
    # 新しい配列を作成して、選ばれたインデックスの位置だけ1にする
    new_arr = [0] * len(example["labels"])
    new_arr[most_freq_label] = 1
    example["labels"] = new_arr
    
    return example