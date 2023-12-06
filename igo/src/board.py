import matplotlib.pyplot as plt
from sgfmill import sgf
from sgfmill import boards


def get_board_at_move(game, move_number):
    # ゲームの初期盤面を取得
    board_size = game.get_size()
    board = boards.Board(board_size)

    # 指定された手数までの手を再生して盤面を更新
    main_sequence = game.get_main_sequence()
    for i, node in enumerate(main_sequence):
        if i > move_number:
            break
        color, move = node.get_move()
        if move is not None:
            x, y = move
            board.play(x, y, color)

    # 白石と黒石の配置を格納する行列
    white_matrix = [[0 for _ in range(board_size)] for _ in range(board_size)]
    black_matrix = [[0 for _ in range(board_size)] for _ in range(board_size)]

    for x in range(board_size):
        for y in range(board_size):
            stone = board.get(x, y)
            if stone == 'b': black_matrix[y][x] = 1  # 黒石を記録
            elif stone == 'w': white_matrix[y][x] = 1  # 白石を記録

    return black_matrix, white_matrix



def visualize_board_at_move(game, move_number):
    # ゲームの初期盤面を取得
    board_size = game.get_size()
    board = boards.Board(board_size)

    # 指定された手数までの手を再生して盤面を更新
    main_sequence = game.get_main_sequence()
    for i, node in enumerate(main_sequence):
        if i > move_number:
            break
        color, move = node.get_move()
        if move is not None:
            x, y = move
            board.play(x, y, color)

    # 盤面を可視化
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    # 枠線を盤面の端の罫線に合わせる
    ax.set_xlim([-0.5, board_size - 0.5])
    ax.set_ylim([-0.5, board_size - 0.5])

    # 罫線を描画
    for i in range(board_size):
        ax.plot([i, i], [0, board_size-1], color='grey')
        ax.plot([0, board_size-1], [i, i], color='grey')

    # 目盛りを削除
    ax.set_xticks([])
    ax.set_yticks([])

    # 石を描画し、行列に記録
    stone_size = 13
    for x in range(board_size):
        for y in range(board_size):
            stone = board.get(x, y)
            if stone == 'b':
                ax.plot(x, y, 'o', color='black', markersize=stone_size)
            elif stone == 'w':
                ax.plot(x, y, 'o', color='white', markeredgecolor='black', markersize=stone_size)

    plt.show()
    return 0


def visualize_matrix(black_matrix, white_matrix, board_size=19):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    # 枠線を盤面の端の罫線に合わせる
    ax.set_xlim([-0.5, board_size - 0.5])
    ax.set_ylim([-0.5, board_size - 0.5])

    # 罫線を描画
    for i in range(board_size):
        ax.plot([i, i], [0, board_size-1], color='grey')
        ax.plot([0, board_size-1], [i, i], color='grey')

    # 目盛りを削除
    ax.set_xticks([])
    ax.set_yticks([])

    # 石を描画し、行列に記録
    stone_size = 13
    for x in range(board_size):
        for y in range(board_size):
            if black_matrix[y][x] == 1:
                ax.plot(x, y, 'o', color='black', markersize=stone_size)
            elif white_matrix[y][x] == 1:
                ax.plot(x, y, 'o', color='white', markeredgecolor='black', markersize=stone_size)

    plt.show()
    return 0


