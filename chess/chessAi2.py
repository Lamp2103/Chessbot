from tensorflow.keras.models import load_model
import numpy as np
import time

class ChessAI:
    def __init__(self, model_path='chess_model_tf.h5'):
        self.model = load_model(model_path)
    
    def board_to_input(self, board):
        input_array = np.zeros((8, 8, 12))  # 6 quân trắng + 6 quân đen

        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }

        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece != '--':
                    key = piece[1] if piece[0] == 'w' else piece[1].lower()
                    index = piece_map.get(key)
                    if index is not None:
                        input_array[row, col, index] = 1
        return np.expand_dims(input_array, axis=0)

    
    def evaluate_board(self, board):
        board_input = self.board_to_input(board)
        return self.model.predict(board_input)[0][0]


    def find_best_move(self, game_state, max_time=10.0, depth=4):
        start_time = time.time()
        best_move = None

        while True:
            if time.time() - start_time >= max_time:
                break

            move, completed = self._deep_search(game_state, depth, start_time, max_time)
            if completed:
                best_move = move
                depth += 1
            else:
                break

        return best_move

    def _deep_search(self, game_state, depth, start_time, max_time):
        best_move = None
        max_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for move in game_state.getValidMoves():
            if time.time() - start_time >= max_time:
                return best_move, False  # Dừng sớm

            # Lưu color trước khi đổi lượt
            color = 1 if game_state.whiteToMove else -1

            game_state.makeMove(move)
            score = -self.negamax(game_state, depth - 1, -beta, -alpha, -color, start_time, max_time)
            game_state.undoMove()

            if score > max_score:
                max_score = score
                best_move = move
            alpha = max(alpha, score)

        return best_move, True

    def negamax(self, game_state, depth, alpha, beta, color, start_time, max_time):
        if time.time() - start_time >= max_time:
            return 0  # Trả về giá trị trung lập nếu hết giờ

        if depth == 0 or game_state.checkmate or game_state.stalemate:
            return color * self.evaluate_board(game_state.board)

        max_score = -float('inf')
        for move in game_state.getValidMoves():
            game_state.makeMove(move)
            score = -self.negamax(game_state, depth - 1, -beta, -alpha, -color, start_time, max_time)
            game_state.undoMove()

            if score > max_score:
                max_score = score
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Cắt tỉa

        return max_score


