import numpy as np

def label_count(example):
    # ラベルの個数をカウントするための辞書 
    label_counts = {}
    # 各ラベルセットに対してループ
    for data in example:
        labels = data["labels"]
        indices = np.nonzero(labels)[0]
        for index in indices:
            if index not in label_counts:
                label_counts[index] = 1
            label_counts[index] += 1
    
    # 辞書を値でソート
    sorted_label_counts = dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True))
    
    # ソートされたラベルインデックス
    sorted_label_keys = list(sorted_label_counts.keys())
    
    
    return sorted_label_keys