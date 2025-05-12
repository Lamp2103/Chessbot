import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Tải dữ liệu đã lưu
X_data = np.load("X_data.npy")
y_data = np.load("y_data.npy")

# Các tham số
input_shape = (8, 8, 12)
num_classes = 4096  # 64x64 nước đi có thể

# Tạo tập dữ liệu
from prepareDataset import create_tf_dataset
dataset = create_tf_dataset(X_data, y_data, batch_size=64)

# Xây dựng mô hình CNN
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Khởi tạo mô hình
model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(dataset, epochs=500)

# Lưu mô hình đã huấn luyện
model.save("chess_model_tf.h5")
print("\nModel saved to chess_model_tf.h5")
