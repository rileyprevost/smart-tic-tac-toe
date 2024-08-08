import numpy as np
def pick_empty_pos(board, board_null, pos):
    """ finds first empty position in board from a given list of positions

    Args:
        board (numpy.array): a square matrix of the current gameboard
        board_null (int): the value representing an empty spot in the gameboard
        pos (list): list of positions (tuples)
    Returns:
        (tuple): empty position
    """
    for p in pos:
        # position is empty
        if board[p] == board_null:
            return p

    # as a fail safe, if there is no empty position in pos:
    empty_pos = np.where(board == board_null)
    return empty_pos[0][0], empty_pos[1][0]

def get_win_set_type(board, player_idx):
    """ determines whether a given player has won and which type of win they have

        win_types: 'row', 'col', 'main_diag', 'other_diag'

    Args:
        board (np.array): a square tic-tac-toe board
        player_idx (int): value of player we're interested in
    Returns:
        win_set (set): set of player indicies that have won; should always be of length 1
        win_type (tuple): indicates whether player won in a row, column
                main diagonal, other diagonal and index of row / column if applicable
    """
    win_set = set()
    win_type = ('', None)

    # assuming board is a square matrix, m = n, so which one
    # we iterate over is arbitrary
    for idx in range(board.shape[0]):
        # current row
        row = board[idx, :]
        # current col
        col = board[:, idx]

        row_set = set(row)
        col_set = set(col)

        player_idx_set = {player_idx}

        # checking for win in rows, casting to set removes duplicate values,
        # so a win would have a win_set length of 1
        if len(row_set) == 1 and row_set == player_idx_set:
            # since all values in the array are equal, we can just add some abritary value from it
            win_set.add(row[0])
            # add win type
            win_type = ('row', idx)
        # checking for win in cols
        if len(col_set) == 1 and col_set == player_idx_set:
            # since all values in the array are equal, we can just add some abritary value from it
            win_set.add(col[0])
            # add win type
            win_type = ('col', idx)

    # get the diagonals
    main_diag = board.diagonal()
    other_diag = board[:, ::-1].diagonal()  # reversing matrix then getting new main diagonal

    main_diag_set = set(main_diag)
    other_diag_set = set(other_diag)

    # check for wins in diagonals
    if len(main_diag_set) == 1 and main_diag_set == player_idx_set:
        # can add some arbitrary value from main_diag as all vals are the same
        win_set.add(main_diag[0])
        # add win type
        win_type = ('main_diag', None)
    # win on other diagonal
    if len(set(other_diag)) == 1 and other_diag_set == player_idx_set:
        # can add some arbitrary value from other_diag as all vals are the same
        win_set.add(other_diag[0])
        # add win type
        win_type = ('other_diag', None)

    return win_set, win_type


def get_win_move(board, board_null, win_type):
    """ finds the winning move for a tic-tac-toe game

        if the game is one move away from winning, winning move will be found

    Args:
        board (numpy.array): a square matrix of the current gameboard
        board_null (int): the value representing an empty spot in the gameboard
        win_type (tuple): indicates whether player won in a row, column
            main diagonal, other diagonal and index of row / column if applicable
    """
    poss_win_moves = []

    if win_type[0] == 'row':
        # in this scenario, we have 2 possible win positions.
        # We know the board is either:
        # [[1,1,0],
        #  [0,0,0],
        #  [0,0,0]]
        # OR
        # [[0,1,1],
        #  [0,0,0],
        #  [0,0,0]]

        # So, the winning move is either in the first or last positions of the current row
        poss_win_moves = [(win_type[1], i) for i in range(board.shape[0])]
    elif win_type[0] == 'col':
        # the situation is similar to the row case, but just in terms of columns
        # indices will flip
        poss_win_moves = [(i, win_type[1]) for i in range(board.shape[0])]
    elif win_type[0] == 'main_diag':
        # Scenarios:
        # [[1,0,0],
        #  [0,1,0],
        #  [0,0,0]]
        # OR
        # [[0,0,0],
        #  [0,1,0],
        #  [0,0,1]]

        poss_win_moves = [(i, i) for i in range(board.shape[0])]
    elif win_type[0] == 'other_diag':
        # Scenarios:
        # [[0,0,1],
        #  [0,1,0],
        #  [0,0,0]]
        # OR
        # [[0,0,0],
        #  [0,1,0],
        #  [1,0,0]]
        # 2,0; 1,1; 0,2 

        poss_win_moves = [(2,0), (1,1), (0,2)]

    # return empty position of possible win moves list
    return pick_empty_pos(board, board_null, poss_win_moves)


def one_move_from_win(board, board_null, player_idx):
    """ finds out if a player is 1 move away from winning a game of tic-tac-toe

    Args:
        board (numpy.array): a square matrix of the current gameboard
        board_null (int): the value representing an empty spot in the gameboard
        player_idx (int): the value the current player is playing with
    Returns:
        (boolean): True if player is one move away from winning, False if not
        win_pos (tuple): position of winning next move
    """
    # we can split up the square matrix (n by n) board into 4 (n-1) by (n-1) matrices
    # from there, we can check if there was a win in any of those matrices, if there was,
    # then the player is 1 move away from winning in the n by n board.

    wins = set()
    win_pos = []

    # Ex: 3x3 matrix
    # [[0,1,2],
    #  [3,4,5],
    #  [6,7,8]]

    # this gives us:
    # [[4,5],
    #  [7,8]]
    sub_mat_1 = board[1:, 1:]

    # get win set and win positions
    win_info = get_win_set_type(sub_mat_1, player_idx)
    wins = wins | win_info[0]
    win_pos = get_win_move(board, board_null, win_info[1])

    # this gives us:
    # [[0,1],
    #  [3,4]]
    sub_mat_2 = board[0:-1, 0:-1]

    # get win set and win positions
    win_info = get_win_set_type(sub_mat_2, player_idx)
    wins = wins | win_info[0]
    win_pos = get_win_move(board, board_null, win_info[1])

    # this gives us:
    # [[3,4],
    #  [6,7]]
    sub_mat_3 = board[1:, :-1]

    # get win set and win positions
    win_info = get_win_set_type(sub_mat_3, player_idx)
    wins = wins | win_info[0]
    win_pos = get_win_move(board, board_null, win_info[1])

    # this gives us:
    # [[1,2],
    #  [4,5]]
    sub_mat_4 = board[:-1, 1:]

    # get win set and win positions
    win_info = get_win_set_type(sub_mat_4, player_idx)
    wins = wins | win_info[0]
    win_pos = get_win_move(board, board_null, win_info[1])

    # if the wins set is empty, we found no winners in the 4 sub-matrices
    if wins != set():
        return True, win_pos
    else:
        return False, win_pos


def find_decent_move(board, board_null, player_idx):
    """ finds a decent move for the computer in a game of tic-tac-toe

        either the computer is one move from winning and makes that move or
        fills an empty position.

    Args:
        board (numpy.array): a square matrix of the current gameboard
        board_null (int): the value representing an empty spot in the gameboard
        player_idx (int): the value the user is playing with (1 or 2)
    Returns:
        (int): row_idx on the board of decent next move
        (int): col_idx on the board of decent next move
    """
    win_info = one_move_from_win(board, board_null, player_idx)

    # check if player is 1 move from win
    if win_info[0]:
        # return win position
        return win_info[1]
    # else just pick first empty position
    else:
        # getting all positions that are empty in board
        empty_pos = np.where(board == board_null)

        # return first empty position
        return empty_pos[0][0], empty_pos[1][0]


def get_position(player_idx):
    """ gets a user's move via input().  no user input validation

    see also: get_apply_input()

    Args:
        player_idx (int): player idx (used to call
            the player by name)

    Returns:
        row_idx (int): a row index
        col_idx (int): a col index
    """
    # get input from user
    pos = input(f'player{player_idx} input position: ')

    # parse user's input
    row_idx, col_idx = pos.split(',')
    row_idx = int(row_idx)
    col_idx = int(col_idx)

    return row_idx, col_idx

def get_apply_input(board, player_idx, board_null=0):
    """ gets input from user and applies their mark

    re-query if input given does not refer to a
    position on the board currently marked as board_null

    Args:
        board (np.array): a 3x3 tic-tac-toe board
        player_idx (int): player whose turn is being taken
            (either 1 or 2)
        board_null (int): the value of open positions
            on the board

    Returns:
        board (np.array): a 3x3 tic-tac-toe board
            which has recorded
    """
    assert player_idx in (1, 2), 'invalid player_idx'

    while True:
        # get user input
        row_idx, col_idx = get_position(player_idx)

        # if position given is already taken, requery
        if board[row_idx, col_idx] != board_null:
            print("Invalid input given, position already taken.")
            continue

        # if position is 'empty', stop querying
        if board[row_idx, col_idx] == board_null:
            break

    # updating the board
    board[row_idx, col_idx] = player_idx

    return board


def play_tic_tac_toe_xc(board_null=0, shape=(3, 3)):
    """ plays a game of tic-tac-toe on a square matrix board.

        One player playing against a semi-intelligent computer.

    Args:
        board_null (int): null value in board (spaces which
            player may select)
        shape (tuple): a tuple of length 2, num
            rows and num cols of tic-tac-toe board
    Returns:
        None.
    """

    # initialize board
    board = np.full(shape, board_null)

    # display original board
    print(board)

    while True:
        # computer makes move first -- designate index as 1
        print("Computer (Player1) makes move:")

        # see if opponent is 1 move from winning and get position to counteract that
        opp_win_info = one_move_from_win(board, board_null, 2)

        # if player2 is one move away from winning:
        if opp_win_info[0]:
            pos = opp_win_info[1]
        # if not, pick random empty position or win position
        else:
            pos = find_decent_move(board, board_null, 1)

        # update board with computer move
        board[pos] = 1

        # display board
        print(board)

        win_set = get_win_set_type(board, 1)[0]

        # check for win after each move
        if 1 in win_set or 2 in win_set:
            break

        # check for full board (tie)
        if board_null not in board:
            break

        # player2 makes move
        get_apply_input(board, 2, board_null=board_null)

        # display board
        print(board)

        win_set = get_win_set_type(board, 2)[0]

        # check for win after each move
        if 1 in win_set or 2 in win_set:
            break

        # check for full board (tie)
        if board_null not in board:
            break

    # remove board_null from win_set if in it
    if board_null in win_set:
        win_set.remove(board_null)

    # game outcome
    if win_set != set():
        print(f"Player{win_set} has won!")
    else:
        print("Tie!")

play_tic_tac_toe_xc()