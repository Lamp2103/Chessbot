import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorflow.keras.models import load_model
import chess.pgn
import os

def board_to_tensor(board: chess.Board):
    # 12 channels: [wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK]
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    tensor = torch.zeros(12, 8, 8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - square // 8
            col = square % 8
            idx = piece_map[piece.symbol()]
            tensor[idx][row][col] = 1
    return tensor

def extract_training_data_from_pgn(pgn_file_path, max_games=None):
    X = []
    y = []

    with open(pgn_file_path) as pgn:
        count = 0
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None or (max_games and count >= max_games):
                break

            board = game.board()
            result = game.headers["Result"]
            # Gán nhãn kết quả: thắng trắng: 1, hòa: 0, thắng đen: -1
            result_value = 1 if result == "1-0" else -1 if result == "0-1" else 0

            for move in game.mainline_moves():
                board_tensor = board_to_tensor(board)
                X.append(board_tensor)
                y.append(torch.tensor([result_value], dtype=torch.float))
                board.push(move)

            count += 1
            if count % 100 == 0:
                print(f"Processed {count} games.")

    return X, y


# Neural Network model
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Output: position evaluation

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Convert board to tensor
PIECE_TO_PLANE = {
    'wK': 0, 'wQ': 1, 'wR': 2, 'wB': 3, 'wN': 4, 'wp': 5,
    'bK': 6, 'bQ': 7, 'bR': 8, 'bB': 9, 'bN': 10, 'bp': 11
}

def board_to_tensor(gs):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            piece = gs.board[r][c]
            if piece != "--":
                plane = PIECE_TO_PLANE[piece]
                tensor[plane][r][c] = 1.0
    return tensor.flatten()

# Training function
def train(model, optimizer, criterion, game_data, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for gs, value in game_data:
            x = board_to_tensor(gs)
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 768]
            y_tensor = torch.tensor([[value]], dtype=torch.float32)      # Shape: [1, 1]

            optimizer.zero_grad()
            output = model(x_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Example: using dummy data for testing
def dummy_game_state():
    class DummyGS:
        def __init__(self):
            self.board = [["--" for _ in range(8)] for _ in range(8)]
            self.board[0][0] = "wR"
            self.board[7][7] = "bK"
            self.whiteToMove = True
    return DummyGS()

if __name__ == "__main__":
    model = ChessNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Generate fake data: [(GameState, value)]
    dummy_data = [(dummy_game_state(), 0.5) for _ in range(100)]
    train(model, optimizer, criterion, dummy_data, epochs=10)

    torch.save(model.state_dict(), "chess_model.pth")
