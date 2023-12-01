import os
import glob

from tqdm import tqdm

import cshogi
from sklearn.model_selection import train_test_split

csa_file_list = glob.glob(os.path.join("..", "raw_data","*","*.csa"))

filtered_list = []

kif_num = 0
position_num = 0
filter_moves = 50

board = cshogi.Board()

for file_path in tqdm(csa_file_list):
    for kif in cshogi.Parser.parse_file(file_path):
        if len(kif.moves) < filter_moves:
            continue

        board.set_sfen(kif.sfen)
        try:
            for move_i, move in enumerate(kif.moves):
                if not board.is_legal(move):
                    raise Exception()
                board.push(move)
        except:
            print(f"skip {file_path}")
            continue
        
        filtered_list.append(file_path)
        kif_num += 1
        position_num += len(kif.moves)

print("kif_num", kif_num)
print("position_num", position_num)

train_file_list, test_file_list = train_test_split(filtered_list, test_size=0.05)

with open("../preprocessed_data/train_file_path.txt", "w") as f:
    for train_file in train_file_list:
        f.write("%s\n" % train_file)

with open("../preprocessed_data/test_file_path.txt", "w") as f:
    for test_file in test_file_list:
        f.write("%s\n" % test_file)