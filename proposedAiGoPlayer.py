import numpy as np
import copy
import time
from collections import deque, defaultdict

# Game Constants
BOARD_SIZE = 5
EMPTY, BLACK, WHITE = 0, 1, 2
MAX_DEPTH = 4
TIME_LIMIT = 7.5
KOMI = 2.5
KO_HISTORY_LENGTH = 3

# Strategic Position Weights
POSITION_WEIGHTS = [
    [1.4, 0.9, 1.0, 0.9, 1.4],
    [0.9, 1.0, 0.8, 1.0, 0.9],
    [1.0, 0.8, 1.41, 0.8, 1.0],
    [0.9, 1.0, 0.8, 1.0, 0.9],
    [1.4, 0.9, 1.0, 0.9, 1.4]
]

class EnhancedGoAgent:
    def __init__(self):
        self.move_counter = 0
        self.ko_history = deque(maxlen=KO_HISTORY_LENGTH)
        self.pattern_bonus = {
            'capture': 1.8,  # Moderate capture bonus
            'center_control': 1.0,  # Strong center control bonus
            'liberty_expansion': 1.2,  # Bonus for expanding liberties
            'strong_connection': 1.27,  # Bonus for strong group connections
            'weak_group': -2.0,  # Penalty for weak groups
        }

    def read_input(self):
        """Read input.txt and parse game state"""
        with open("input.txt", "r") as f:
            lines = f.readlines()
            stone = int(lines[0].strip())
            prev_board = [[int(x) for x in line.strip()] for line in lines[1:6]]
            curr_board = [[int(x) for x in line.strip()] for line in lines[6:11]]
        return stone, prev_board, curr_board

    def write_output(self, action):
        """Write move to output.txt"""
        with open("output.txt", "w") as f:
            f.write("PASS" if action == "PASS" else f"{action[0]},{action[1]}")

    def find_liberties(self, board, row, col, stone_type):
        """Calculate liberties for a stone group and return the set of liberty positions"""
        visited = set()
        liberties = set()
        stack = [(row, col)]

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
        return liberties

    def find_dead_stones(self, board, stone_type):
        """Identify stones with no liberties."""
        dead = []
        visited = set()

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == stone_type and (i, j) not in visited:
                    liberties = self.find_liberties(board, i, j, stone_type)
                    if not liberties:
                        # Find the entire group and mark all stones as dead
                        cluster = self.find_cluster(board, i, j, stone_type)
                        dead.extend(cluster)
                        visited.update(cluster)
        return dead

    def remove_stones(self, board, stones):
        """Remove stones from board"""
        new_board = copy.deepcopy(board)
        for i, j in stones:
            new_board[i][j] = EMPTY
        return new_board

    def find_cluster(self, board, row, col, stone_type):
        """Find all stones in a connected cluster using DFS."""
        cluster = []
        stack = [(row, col)]
        visited = set()

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            cluster.append((r, c))

            # Check all four directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if board[nr][nc] == stone_type and (nr, nc) not in visited:
                        stack.append((nr, nc))
        return cluster

    def is_valid_move(self, board, move, stone_type):
        """Check move validity including suicide rule with capture exception."""
        if move == "PASS":
            return True

        i, j = move
        if board[i][j] != EMPTY:
            return False

        # Simulate move
        new_board = copy.deepcopy(board)
        new_board[i][j] = stone_type

        # Remove opponent dead stones
        dead_opponent = self.find_dead_stones(new_board, 3 - stone_type)
        new_board = self.remove_stones(new_board, dead_opponent)

        # Check if the move captures any opponent stones
        if dead_opponent:
            # If it captures, it's valid regardless of liberties
            return True

        # Check suicide rule if no capture
        # Remove your own dead stones (if any)
        dead_self = self.find_dead_stones(new_board, stone_type)
        if dead_self:
            # If the move results in your own stones being captured, it's invalid
            return False

        # Check if the move has at least one liberty
        return len(self.find_liberties(new_board, i, j, stone_type)) > 0

    def prioritize_moves(self, board, stone_type):
        """Prioritize moves that strengthen position and control key areas"""
        moves = []

        # Strategic points (center and corners)
        strategic_points = [(2, 2), (1, 1), (1, 3), (3, 1), (3, 3), (0, 0), (0, 4), (4, 0), (4, 4)]
        moves.extend(p for p in strategic_points if board[p[0]][p[1]] == EMPTY)

        # Liberty expansion moves
        liberty_moves = self.find_liberty_expansions(board, stone_type)
        moves.extend(liberty_moves)

        # Capture opportunities (only if safe)
        safe_captures = self.find_safe_capture_moves(board, stone_type)
        moves.extend(safe_captures)

        return list(dict.fromkeys(moves))  # Remove duplicates while preserving order

    def find_liberty_expansions(self, board, stone_type):
        """Find moves that expand liberties for multiple groups"""
        liberty_map = defaultdict(int)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == stone_type:
                    libs = self.find_liberties(board, i, j, stone_type)
                    for l in libs:
                        liberty_map[l] += 1
        return [pos for pos, count in sorted(liberty_map.items(), key=lambda x: -x[1])]

    def find_safe_capture_moves(self, board, stone_type):
        """Find capture moves that don't leave weak groups."""
        captures = []
        opponent = 3 - stone_type
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == opponent:
                    libs = self.find_liberties(board, i, j, opponent)
                    if len(libs) == 1:
                        capture_move = libs.pop()
                        # Check if the capture move doesn't leave us weak
                        new_board = self.simulate_move(board, capture_move, stone_type)
                        if not self.is_weak_group(new_board, capture_move[0], capture_move[1], stone_type):
                            captures.append(capture_move)
        return captures

    def is_weak_group(self, board, i, j, stone_type):
        """Check if a group is weak (<= 1 liberty)"""
        libs = self.find_liberties(board, i, j, stone_type)
        return len(libs) <= 1

    def simulate_move(self, board, move, stone_type):
        """Apply move and return new board state."""
        if move == "PASS":
            return copy.deepcopy(board)

        new_board = copy.deepcopy(board)
        i, j = move
        new_board[i][j] = stone_type

        # Remove opponent dead stones
        dead_opponent = self.find_dead_stones(new_board, 3 - stone_type)
        new_board = self.remove_stones(new_board, dead_opponent)

        # Remove your own dead stones (if any)
        dead_self = self.find_dead_stones(new_board, stone_type)
        new_board = self.remove_stones(new_board, dead_self)

        return new_board

    def is_ko(self, new_board):
        """Check if the new board state is a Ko."""
        return any(np.array_equal(new_board, state) for state in self.ko_history)

    def calculate_liberties(self, board, stone_type):
        """Calculate total liberties for all groups of the given stone type."""
        total_liberties = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == stone_type:
                    liberties = self.find_liberties(board, i, j, stone_type)
                    total_liberties += len(liberties)
        return total_liberties

    def detect_weak_groups(self, board, stone_type):
        """Identify groups with 1-2 liberties."""
        weak = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == stone_type:
                    libs = self.find_liberties(board, i, j, stone_type)
                    if 1 <= len(libs) <= 2:
                        weak.append((i, j))
        return weak

    def enhanced_evaluate(self, board, stone_type):
        """Evaluate the board with a focus on long-term positional strength."""
        score = 0
        opponent = 3 - stone_type

        # Material and positional advantage
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == stone_type:
                    score += POSITION_WEIGHTS[i][j]
                    # Bonus for strong connections
                    score += 0.3 * (self.count_adjacent_friends(board, i, j) - 1)
                elif board[i][j] == opponent:
                    score -= POSITION_WEIGHTS[i][j] * 0.9

        # Liberty dominance
        my_libs = self.calculate_liberties(board, stone_type)
        opp_libs = self.calculate_liberties(board, opponent)
        score += 2.0 * (my_libs - opp_libs) * self.pattern_bonus['liberty_expansion']

        # Weak group penalty
        weak_penalty = len(self.detect_weak_groups(board, stone_type)) * 1.5
        score -= weak_penalty

        # Bonus for capturing moves
        dead_opponent = self.find_dead_stones(board, opponent)
        if stone_type == BLACK:
            # Boost capture bonus for Black to offset Komi
            score += len(dead_opponent) * self.pattern_bonus['capture'] * 2.4  # Higher bonus for Black
        else:
            score += len(dead_opponent) * self.pattern_bonus['capture']

        # Bonus for capturing more than 4 stones
        if len(dead_opponent) > 4:
            score += len(dead_opponent) * self.pattern_bonus['capture'] * 5.0  # Higher bonus for large captures
      
        # Adjust for komi (only when playing as Black)
        if stone_type == BLACK:
            score -= KOMI  # Subtract Komi for Black
        elif stone_type == WHITE:
            score += KOMI  # Add Komi for White

        return score

    def count_adjacent_friends(self, board, i, j):
        stone = board[i][j]
        return sum(1 for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                   if 0 <= i + di < 5 and 0 <= j + dj < 5 and board[i + di][j + dj] == stone)

    def dynamic_alpha_beta(self, board, depth, alpha, beta, maximizing, stone_type, start_time):
        """Alpha-beta pruning with time management"""
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError()

        if depth == 0:
            return self.enhanced_evaluate(board, stone_type)

        moves = self.prioritize_moves(board, stone_type)

        if maximizing:
            value = -np.inf
            for move in moves:
                new_board = self.simulate_move(board, move, stone_type)
                if self.is_ko(new_board):
                    continue
                value = max(value, self.dynamic_alpha_beta(new_board, depth - 1, alpha, beta, False, stone_type, start_time))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = np.inf
            for move in moves:
                new_board = self.simulate_move(board, move, 3 - stone_type)
                if self.is_ko(new_board):
                    continue
                value = min(value, self.dynamic_alpha_beta(new_board, depth - 1, alpha, beta, True, stone_type, start_time))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
    def endgame_evaluation(self, board, stone_type):
        """Evaluate the board in the endgame phase."""
        score = 0
        opponent = 3 - stone_type

        # Territory counting
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == EMPTY:
                    # Check if the empty intersection is likely to belong to a player
                    if self.is_territory(board, i, j, stone_type):
                        score += 1
                    elif self.is_territory(board, i, j, opponent):
                        score -= 1

        # Adjust for Komi
        if stone_type == BLACK:
            score -= KOMI
        else:
            score += KOMI

        return score

    def is_territory(self, board, i, j, stone_type):
        """Check if an empty intersection is likely to belong to a player."""
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                if board[ni][nj] == 3 - stone_type:
                    return False
        return True
    def choose_action(self):
        """Main decision function with time management"""
        stone_type, prev_board, curr_board = self.read_input()
        self.move_counter += 1
        self.ko_history.append(prev_board)
        start_time = time.time()
        best_move = "PASS"
        best_value = -np.inf

        try:
            for depth in range(2, MAX_DEPTH + 1):
                current_best = None
                current_val = -np.inf

                for move in self.prioritize_moves(curr_board, stone_type):
                    if time.time() - start_time > TIME_LIMIT:
                        raise TimeoutError()

                    if not self.is_valid_move(curr_board, move, stone_type):
                        continue

                    new_board = self.simulate_move(curr_board, move, stone_type)
                    if self.is_ko(new_board):
                        continue

                    eval = self.dynamic_alpha_beta(
                        new_board, depth - 1, -np.inf, np.inf,
                        False, stone_type, start_time
                    )
                    if self.move_counter >= 20:  # Endgame phase
                        eval += self.endgame_evaluation(new_board, stone_type)
                    

                    if eval > current_val:
                        current_val = eval
                        current_best = move

                if current_val > best_value:
                    best_value = current_val
                    best_move = current_best

        except TimeoutError:
            pass

        self.write_output(best_move if best_move else "PASS")

if __name__ == "__main__":
    agent = EnhancedGoAgent()
    agent.choose_action()