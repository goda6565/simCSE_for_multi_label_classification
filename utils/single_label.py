import numpy as np

# kNNに通すデータをシングルラベルに変更

def random_choice(example):
    # 1のインデックスを取得
    indices_of_ones = [i for i, x in enumerate(example["labels"]) if x == 1]

    # ランダムに1つのインデックスを選択
    chosen_index = np.random.choice(indices_of_ones)

    # 新しい配列を作成して、選ばれたインデックスの位置だけ1にする
    new_arr = [0] * len(example["labels"])
    new_arr[chosen_index] = 1
    example["labels"] = new_arr
    return example