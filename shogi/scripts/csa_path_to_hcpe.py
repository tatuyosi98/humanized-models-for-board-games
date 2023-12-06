import numpy as np

from tqdm import tqdm

import cshogi
import cshogi.CSA


train_path = "../preprocessed_data/train_file_path.txt"
test_path = "../preprocessed_data/test_file_path.txt"

train_f = open("../preprocessed_data/train.hcpe", "wb")
test_f = open("../preprocessed_data/test.hcpe", "wb")

hcpes = np.zeros(1024, cshogi.HuffmanCodedPosAndEval)

board = cshogi.Board()
for path_file, f in zip([train_path, test_path], [train_f, test_f]):
    kif_num = 0
    position_num = 0
    with open(path_file, "r") as csa_path_f:
        csa_path_list = [string.rstrip() for string in csa_path_f.readlines()]
    for csa_path in tqdm(csa_path_list):
        # print(csa_path)
        for kif in cshogi.CSA.Parser.parse_file(csa_path):
            board.set_sfen(kif.sfen)
            p = 0
            for move in kif.moves:
                hcpe = hcpes[p]
                p += 1

                # 局面をhcpに変換
                board.to_hcp(hcpe["hcp"])
                # 指し手の32bit数値を16bitに切り捨てる
                hcpe["bestMove16"] = cshogi.move16(move)
                # 勝敗結果
                hcpe["gameResult"] = kif.win

                board.push(move)
            
            hcpes[:p].tofile(f)

            kif_num += 1
            position_num += p

    print("kif_num", kif_num)
    print("position_num", position_num)
