from cshogi import *

# 移動方向を表す定数
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT,
    UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
] = range(20)

# 入力特徴量の数
FEATURES_NUM = len(PIECE_TYPES) * 2 + sum(MAX_PIECES_IN_HAND) * 2

# 移動を表すラベルの数
MOVE_PLANES_NUM = len(MOVE_DIRECTION) + len(HAND_PIECES)
MOVE_LABELS_NUM = MOVE_PLANES_NUM * 81

# 入力特徴量を作成
def make_input_features(board, features):
    # featuresのshapeはFEATURES_NUM x 9 x 9

    # 入力特徴量を0に初期化
    features.fill(0)

    # 盤上の駒（歩，と，香，成香，桂，成桂，銀，成銀，金，角，馬，飛，竜，玉）*2 = 28
    if board.turn == BLACK:
        board.piece_planes(features)
        pieces_in_hand = board.pieces_in_hand
    else:
        board.piece_planes_rotate(features)
        pieces_in_hand = reversed(board.pieces_in_hand)
    # 持ち駒
    i = 28
    for hands in pieces_in_hand:
        for num, max_num in zip(hands, MAX_PIECES_IN_HAND):
            features[i:i+num].fill(1)
            i += max_num

# 移動を表すラベルを作成
def make_move_label(move, color):
    if not move_is_drop(move):  # 駒の移動
        to_sq = move_to(move)
        from_sq = move_from(move)

        # 後手の場合盤を回転
        if color == WHITE:
            to_sq = 80 - to_sq
            from_sq = 80 - from_sq

        # 移動方向
        to_x, to_y = divmod(to_sq, 9)
        from_x, from_y = divmod(from_sq, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y
        if dir_y < 0:
            if dir_x == 0:
                move_direction = UP
            elif dir_y == -2 and dir_x == -1:
                move_direction = UP2_RIGHT
            elif dir_y == -2 and dir_x == 1:
                move_direction = UP2_LEFT
            elif dir_x < 0:
                move_direction = UP_RIGHT
            else:  # dir_x > 0
                move_direction = UP_LEFT
        elif dir_y == 0:
            if dir_x < 0:
                move_direction = RIGHT
            else:  # dir_x > 0
                move_direction = LEFT
        else:  # dir_y > 0
            if dir_x == 0:
                move_direction = DOWN
            elif dir_x < 0:
                move_direction = DOWN_RIGHT
            else:  # dir_x > 0
                move_direction = DOWN_LEFT

        # 成り
        if move_is_promotion(move):
            move_direction += 10
    else:  # 駒打ち
        to_sq = move_to(move)
        # 後手の場合盤を回転
        if color == WHITE:
            to_sq = 80 - to_sq

        # 駒打ちの移動方向
        move_direction = len(MOVE_DIRECTION) + move_drop_hand_piece(move)

    return move_direction * 81 + to_sq

# 対局結果から報酬を作成
def make_result(game_result, color):
    if color == BLACK:
        if game_result == BLACK_WIN:
            return 1
        if game_result == WHITE_WIN:
            return 0
    else:
        if game_result == BLACK_WIN:
            return 0
        if game_result == WHITE_WIN:
            return 1
    return 0.5

def feature_to_sfen(feature):
    # boardを文字列リストで再現してそれをsfenに直す方針．
    board_string_list = [[None for _ in range(9)] for _ in range(9)]
    hand_piece_list = []
    sfen = ""

    piece_feature_index = 0

    for color in COLORS:
        for piece_i in PIECE_TYPES:
            for x in range(9):
                for y in range(9):
                    if int(feature[piece_feature_index][x][y]) == 1:
                        board_string_list[y][8-x] = PIECE_SYMBOLS[piece_i]
                        if color == BLACK:
                            # 先手なら大文字化
                            board_string_list[y][8-x] = board_string_list[y][8-x].upper()
            piece_feature_index += 1
            

    for color in COLORS:
        for hand_piece_i, hand_piece_symbol in enumerate(HAND_PIECE_SYMBOLS):
            for piece_i in range(MAX_PIECES_IN_HAND[hand_piece_i]):
                if int(feature[piece_feature_index][0][0] == 1):
                    if color == BLACK:
                        hand_piece_list.append(hand_piece_symbol.upper())
                    elif color == WHITE:
                        hand_piece_list.append(hand_piece_symbol)
                piece_feature_index += 1
    from pprint import pprint
    pprint(board_string_list)
    print(hand_piece_list)

    for board_string_row in board_string_list:
        none_cnt = 0
        for board_i, board_string in enumerate(board_string_row):
            if board_string == None:
                none_cnt += 1
                if board_i == 8:
                    sfen += str(none_cnt)
                elif board_string_row[board_i + 1] == None:
                    continue
            else:
                if none_cnt > 0:
                    sfen += str(none_cnt)
                    none_cnt = 0
                sfen += board_string
        sfen += "/"
    sfen = sfen.rstrip("/") + " b "
    if len(hand_piece_list) == 0:
        sfen += "-"
    for hand_piece in hand_piece_list:
        sfen += hand_piece
    sfen += " 1"
    print(sfen)

    return sfen
