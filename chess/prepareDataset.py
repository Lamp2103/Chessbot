import chess
import numpy as np
import tensorflow as tf
import csv

# Chuyển các quân cờ thành các chỉ số tương ứng
PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def board_to_vector(board):
    """Chuyển bàn cờ (chess.Board) thành tensor shape (8, 8, 12)"""
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            index = PIECE_TO_INDEX[piece.symbol()]
            tensor[row, col, index] = 1.0
    return tensor

def move_to_index(move):
    """Chuyển nước đi thành nhãn số hóa"""
    # Gán index cho tất cả các nước đi hợp lệ (4096 khả năng từ 64x64)
    return move.from_square * 64 + move.to_square

def extract_data_from_csv(file_path, max_games=500):
    """Trích xuất dữ liệu từ file CSV, tạo tensor cho bàn cờ và nhãn cho các nước đi"""
    X_data = []
    y_data = []
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            moves = row['moves'].split()  # Tách các nước đi từ cột 'moves'
            board = chess.Board()  # Khởi tạo bàn cờ mới
            game_moves = []  # Danh sách các nước đi hợp lệ

            for move in moves:
                try:
                    move_obj = board.parse_san(move)  # Chuyển chuỗi thành move object
                    if move_obj in board.legal_moves:
                        board.push(move_obj)
                        X_data.append(board_to_vector(board))
                        y_data.append(move_to_index(move_obj))
                except Exception as e:
                    print(f"Đã xảy ra lỗi với nước đi {move}: {e}")
                    continue

            if len(X_data) >= max_games:
                break  # Dừng nếu đã đạt đủ số lượng ván cờ cần thiết

    return np.array(X_data), np.array(y_data)

def create_tf_dataset(X_data, y_data, batch_size=32):
    """Tạo TensorFlow dataset từ dữ liệu X và y"""
    dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data))
    dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    # Đọc và xử lý dữ liệu từ file CSV
    X_data, y_data = extract_data_from_csv("D:/Chess/chess-ai/chess/data/games.csv", max_games=100000)
    
    # Lưu dữ liệu đã xử lý vào các file numpy
    np.save("X_data.npy", X_data)
    np.save("y_data.npy", y_data)
    
    # In ra thông tin về số lượng mẫu đã lưu
    print(f"Saved dataset: {X_data.shape[0]} samples")
