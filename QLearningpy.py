import random
import copy
import json
import numpy as np
from collections import defaultdict

# Constants
BOARD_SIZE = 5
EMPTY = 0
BLACK = 1
WHITE = 2
ACTIONS = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)] + ['PASS']
Q_VALUES_FILE = "D:/projects/q_table.json"

# Q-learning parameters
ALPHA = 0.3  # Learning rate
GAMMA = 0.8  # Discount factor
EPSILON_START =1.0  # Initial exploration rate
EPSILON_END = 0.2  # Final exploration rate
EPSILON_DECAY = 0.1  # Decay rate for epsilon

# Initialize Q-table
Q = defaultdict(lambda: np.zeros(len(ACTIONS)))

# Helper functions from Alpha-Beta code
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
    return legal_moves

# Q-learning functions
def choose_action(state, prev_state, stone_type, epsilon=0.0):
    """Choose an action using the trained Q-values or immediate rewards."""
    legal_moves = find_legal_moves(state, prev_state, stone_type)
    if not legal_moves:
        print("No legal moves available. Returning PASS.")
        return 'PASS'

    state_key = tuple(map(tuple, state))
    q_values = Q[state_key]

    # Check if all Q-values are zero
    if np.all(q_values == 0):
        print("All Q-values are zero. Selecting move based on immediate reward.")
        best_move = None
        best_reward = -float('inf')
        for move in legal_moves:
            row, col = move
            next_board = copy.deepcopy(state)
            next_board[row][col] = stone_type
            dead_stones = find_dead_stones(next_board, 3 - stone_type)
            next_board = remove_dead_stones(next_board, dead_stones)
            reward = calculate_reward(state, next_board, stone_type)
            if reward > best_reward:
                best_reward = reward
                best_move = move
        print(f"Best move based on immediate reward: {best_move}, Reward: {best_reward}")
        return best_move

    # Otherwise, use Q-values to select the best move
    print(f"Q-values for state {state_key}: {q_values}")
    print(f"Legal moves: {legal_moves}")
    # Filter Q-values for legal moves
    legal_q_values = [q_values[ACTIONS.index(move)] for move in legal_moves]
    best_move_index = np.argmax(legal_q_values)
    best_move = legal_moves[best_move_index]
    print(f"Best move based on Q-values: {best_move}")
    return best_move

def update_q_value(state, action, reward, next_state):
    """Update Q-value using the Q-learning rule."""
    state_key = tuple(map(tuple, state))
    next_state_key = tuple(map(tuple, next_state))
    action_index = ACTIONS.index(action)
    max_next_q = np.max(Q[next_state_key])
    Q[state_key][action_index] += ALPHA * (reward + GAMMA * max_next_q - Q[state_key][action_index])

def simulate_game(stone_type, epsilon):
    """Simulate a game and update Q-values."""
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    prev_board = None
    done = False
    move_count = 0

    while not done:
        print(f"\nMove {move_count + 1}: Player {stone_type}'s turn")
        print("Current board state:")
        for row in board:
            print(row)

        if stone_type == BLACK:
            # Agent's turn (uses Q-values)
            action = choose_action(board, prev_board, stone_type, epsilon)
        else:
            # Random opponent's turn
            legal_moves = find_legal_moves(board, prev_board, stone_type)
            if not legal_moves:
                action = 'PASS'
            else:
                action = random.choice(legal_moves)
            print(f"Random opponent chose: {action}")

        if action == 'PASS':
            print("Player chose to PASS. Ending game.")
            done = True
        else:
            row, col = action
            print(f"Player chose to place stone at ({row}, {col})")
            next_board = copy.deepcopy(board)
            next_board[row][col] = stone_type
            dead_stones = find_dead_stones(next_board, 3 - stone_type)
            next_board = remove_dead_stones(next_board, dead_stones)
            reward = calculate_reward(board, next_board, stone_type)
            print(f"Reward for this move: {reward}")
            update_q_value(board, action, reward, next_board)
            prev_board = board
            board = next_board
            stone_type = 3 - stone_type
            move_count += 1

        if move_count >= 100:  # Prevent infinite loops
            print("Max moves reached. Ending game.")
            done = True

    print("\nFinal board state:")
    for row in board:
        print(row)

def is_ko_capture(board, next_board, stone_type):
    """Check if the move captures a stone in a Ko situation."""
    # Check if the move captures exactly one stone
    captured_stones = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == (3 - stone_type) and next_board[row][col] == EMPTY:
                captured_stones += 1
    if captured_stones != 1:
        return False

    # Check if the move creates a Ko situation
    return is_ko_violation(board, next_board)

def calculate_reward(board, next_board, stone_type):
    """Calculate reward based on the game state."""
    reward = 0

    # Reward for capturing opponent stones
    captured_stones = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == (3 - stone_type) and next_board[row][col] == EMPTY:
                captured_stones += 1
    reward += captured_stones * 3.0  # Higher reward for capturing stones

    # Reward for gaining liberties
    current_liberties = 0
    next_liberties = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == stone_type:
                current_liberties += len(find_liberties(board, row, col, stone_type))
            if next_board[row][col] == stone_type:
                next_liberties += len(find_liberties(next_board, row, col, stone_type))
    liberty_diff = next_liberties - current_liberties
    reward += liberty_diff * 1.5  # Higher reward for increasing liberties

    # Penalty for losing liberties
    if liberty_diff < 0:
        reward += liberty_diff * 1.0  # Higher penalty for reducing liberties

    # Bonus for center control
    center_row, center_col = BOARD_SIZE // 2, BOARD_SIZE // 2
    if next_board[center_row][center_col] == stone_type:
        reward += 1.0  # Higher bonus for controlling the center

    # Penalty for edge moves (unless capturing stones)
    if captured_stones == 0:
        for row, col in [(0, 0), (0, 4), (4, 0), (4, 4)]:  # Corners
            if next_board[row][col] == stone_type:
                reward -= 0.5  # Penalize corner moves
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if (row == 0 or row == BOARD_SIZE - 1 or col == 0 or col == BOARD_SIZE - 1) and next_board[row][col] == stone_type:
                    reward -= 0.2  # Penalize edge moves

    # Reward for Ko captures
    if is_ko_capture(board, next_board, stone_type):
        reward += 5.0  # Very high reward for Ko captures

    # Penalty for Ko violations
    if is_ko_violation(board, next_board):
        reward -= 10.0  # Very high penalty for Ko violations

    return reward
# Training loop
def train_agent(num_episodes):
    epsilon = EPSILON_START
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")
        simulate_game(BLACK, epsilon)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Epsilon: {epsilon}")
            save_q_values(Q, Q_VALUES_FILE)
    save_q_values(Q, Q_VALUES_FILE)
    print("Training completed.")

# Save and load Q-values
def save_q_values(q_table, filename):
    with open(filename, 'w') as file:
        json.dump({str(k): v.tolist() for k, v in q_table.items()}, file)

def load_q_values(q_table, filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            for k, v in data.items():
                q_table[eval(k)] = np.array(v)
    except FileNotFoundError:
        print("Q-values file not found. Starting with an empty Q-table.")

# Main function
if __name__ == "__main__":
    load_q_values(Q, Q_VALUES_FILE)
    train_agent(num_episodes=1000)