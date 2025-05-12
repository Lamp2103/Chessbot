import pandas as pd

def read_csv(file_path):
    try:
        # Đọc tệp CSV vào một DataFrame
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Tệp {file_path} không tồn tại.")
        return None
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc tệp: {e}")
        return None

# Ví dụ sử dụng:
csv_data = read_csv("D:\Chess\chess-ai\chess\data\games.csv")
if csv_data is not None:
    print(csv_data.head())  # In 5 dòng đầu tiên của DataFrame
    
