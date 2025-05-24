import copy
import json
from collections import defaultdict

# Constants
BOARD_SIZE = 5
EMPTY = 0
BLACK = 1
WHITE = 2
DEPTH = 3  # Lookahead depth for Alpha-Beta pruning
TRAINING_FILE = "D:/projects/training_data.json"

# Helper functions
def find_liberties(board, row, col, stone_type):
    """Find liberties for a stone or group of stones."""
    liberties = set()
    stack = [(row, col)]
    visited = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if board[nr][nc] == EMPTY:
                    liberties.add((nr, nc))
                elif board[nr][nc] == stone_type:
                    stack.append((nr, nc))
    return list(liberties)

def find_dead_stones(board, stone_type):
    """Find stones with no liberties."""
    dead_stones = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == stone_type and not find_liberties(board, row, col, stone_type):
                dead_stones.append((row, col))
    return dead_stones

def remove_dead_stones(board, dead_stones):
    """Remove dead stones from the board."""
    for stone in dead_stones:
        board[stone[0]][stone[1]] = EMPTY
    return board

def is_ko_violation(prev_board, curr_board):
    """Check if the current move results in a KO violation."""
    return prev_board == curr_board

def find_legal_moves(board, prev_board, stone_type):
    """Find all legal moves for the current player."""
    legal_moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == EMPTY:
                new_board = copy.deepcopy(board)
                new_board[row][col] = stone_type
                dead_stones = find_dead_stones(new_board, 3 - stone_type)
                new_board = remove_dead_stones(new_board, dead_stones)
                if find_liberties(new_board, row, col, stone_type) and not is_ko_violation(prev_board, new_board):
                    legal_moves.append((row, col))
    print(f"Legal moves for player {stone_type}: {legal_moves}")  # Debug print
    return legal_moves

def heuristic_evaluation(board, stone_type):
    """Evaluate the board state using a heuristic function."""
    score = 0

    # Reward for capturing opponent stones
    dead_stones = find_dead_stones(board, 3 - stone_type)
    score += len(dead_stones) * 5  # Higher reward for captures

    # Reward for controlling the center
    center_row, center_col = BOARD_SIZE // 2, BOARD_SIZE // 2
    if board[center_row][center_col] == stone_type:
        score += 3  # Bonus for controlling the center

    # Reward for liberties
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == stone_type:
                liberties = find_liberties(board, row, col, stone_type)
                score += len(liberties) * 1  # Reward for each liberty

    # Penalty for opponent's liberties
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == 3 - stone_type:
                liberties = find_liberties(board, row, col, 3 - stone_type)
                score -= len(liberties) * 0.5  # Penalty for opponent's liberties

    return score

def alpha_beta(board, prev_board, stone_type, depth, alpha, beta, maximizing_player):
    """Alpha-Beta pruning algorithm."""
    if depth == 0:
        return heuristic_evaluation(board, stone_type), None

    legal_moves = find_legal_moves(board, prev_board, stone_type)
    if not legal_moves:
        return heuristic_evaluation(board, stone_type), 'PASS'

    if maximizing_player:
        max_eval = -float('inf')
        best_move = None
        for move in legal_moves:
            row, col = move
            new_board = copy.deepcopy(board)
            new_board[row][col] = stone_type
            dead_stones = find_dead_stones(new_board, 3 - stone_type)
            new_board = remove_dead_stones(new_board, dead_stones)
            eval, _ = alpha_beta(new_board, board, 3 - stone_type, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in legal_moves:
            row, col = move
            new_board = copy.deepcopy(board)
            new_board[row][col] = stone_type
            dead_stones = find_dead_stones(new_board, 3 - stone_type)
            new_board = remove_dead_stones(new_board, dead_stones)
            eval, _ = alpha_beta(new_board, board, 3 - stone_type, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def train_agent():
    """Train the agent using Alpha-Beta pruning and save the results."""
    training_data = defaultdict(dict)
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    prev_board = None
    stone_type = BLACK

    # Simulate training by exploring possible moves
    for _ in range(10000):  # Increase the number of training iterations
        _, best_move = alpha_beta(board, prev_board, stone_type, DEPTH, -float('inf'), float('inf'), True)
        if best_move == 'PASS':
            break
        training_data[str(board)][str(stone_type)] = best_move
        row, col = best_move
        new_board = copy.deepcopy(board)
        new_board[row][col] = stone_type
        dead_stones = find_dead_stones(new_board, 3 - stone_type)
        new_board = remove_dead_stones(new_board, dead_stones)
        prev_board = board
        board = new_board
        stone_type = 3 - stone_type

    # Save training data to a file
    with open(TRAINING_FILE, "w") as file:
        json.dump(training_data, file)
    print("Training completed. Data saved to", TRAINING_FILE)


if __name__ == "__main__":
    train_agent()