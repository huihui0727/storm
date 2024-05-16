import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

# 定义数据目录和图像尺寸
data_dir = 'data/'
image_height = 366
image_width = 366


# 数据加载函数
def load_data(storm_id):
    storm_path = os.path.join(data_dir, storm_id)
    storm_features = []
    storm_images = []
    storm_labels = []
    for file_name in sorted(os.listdir(storm_path)):
        if file_name.endswith('_features.json'):
            features_path = os.path.join(storm_path, file_name)
            with open(features_path, 'r') as f:
                features = json.load(f)
            ocean = int(features['ocean'])
            storm_features.append([features['relative_time'], ocean])
        elif file_name.endswith('.jpg'):
            image_path = os.path.join(storm_path, file_name)
            image = load_img(image_path, target_size=(image_height, image_width))
            image_array = img_to_array(image) / 255.0
            storm_images.append(image_array)
        elif file_name.endswith('_label.json'):
            label_path = os.path.join(storm_path, file_name)
            with open(label_path, 'r') as f:
                label = json.load(f)
            storm_labels.append(label['wind_speed'])
    return np.array(storm_features), np.array(storm_images), np.array(storm_labels)


# 加载所有风暴数据
storm_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
all_features = []
all_images = []
all_labels = []

print("Loading data...")
for storm_id in tqdm(storm_ids, desc="Storm IDs"):
    features, images, labels = load_data(storm_id)
    all_features.append(features)
    all_images.append(images)
    all_labels.append(labels)

all_features = np.concatenate(all_features, axis=0)
all_images = np.concatenate(all_images, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# 数据标准化
print("Scaling data...")
scaler_features = StandardScaler()
scaler_labels = StandardScaler()
scaled_features = scaler_features.fit_transform(all_features)
scaled_labels = scaler_labels.fit_transform(all_labels.reshape(-1, 1))


# 定义滑动窗口
def create_sliding_window_features(features, images, labels, window_size):
    X_features = []
    X_images = []
    y = []
    for i in range(len(features) - window_size):
        X_features.append(features[i:i + window_size])
        X_images.append(images[i + window_size])
        y.append(labels[i + window_size])
    return np.array(X_features), np.array(X_images), np.array(y)


# 定义滑动窗口大小
window_size = 5
print("Creating sliding window features...")
X_features, X_images, y = create_sliding_window_features(scaled_features, all_images, scaled_labels, window_size)

# 划分训练集和测试集
print("Splitting data into train and test sets...")
X_features_train, X_features_test, X_images_train, X_images_test, y_train, y_test = train_test_split(
    X_features, X_images, y, test_size=0.2, random_state=42)

# 调整输入形状
input_shape_features = (X_features_train.shape[1], X_features_train.shape[2])
input_shape_images = (image_height, image_width, 3)


# 构建混合模型
def build_combined_model(input_shape_features, input_shape_images):
    # LSTM部分
    lstm_input = Input(shape=input_shape_features)
    lstm_layer = layers.LSTM(128, return_sequences=True)(lstm_input)
    lstm_layer = layers.LSTM(64)(lstm_layer)
    lstm_output = layers.Dropout(0.5)(lstm_layer)

    # 图片部分
    cnn_input = Input(shape=input_shape_images)
    cnn_layer = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_input)
    cnn_layer = layers.MaxPooling2D((2, 2))(cnn_layer)
    cnn_layer = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cnn_layer)
    cnn_layer = layers.MaxPooling2D((2, 2))(cnn_layer)
    cnn_layer = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(cnn_layer)
    cnn_layer = layers.MaxPooling2D((2, 2))(cnn_layer)
    cnn_layer = layers.Flatten()(cnn_layer)
    cnn_output = layers.Dense(128, activation='relu')(cnn_layer)

    # 合并LSTM和CNN的输出
    combined = layers.concatenate([lstm_output, cnn_output])

    output = layers.Dense(32, activation='relu')(combined)
    output = layers.Dense(1)(output)

    model = Model(inputs=[lstm_input, cnn_input], outputs=output)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    return model


# 构建和训练模型
print("Building and training the model...")
model = build_combined_model(input_shape_features, input_shape_images)
history = model.fit([X_features_train, X_images_train], y_train, epochs=10, batch_size=32, validation_split=0.2)

# 保存模型和标准化器
print("Saving the model and scalers...")
model.save('wind_speed_model.h5')
joblib.dump(scaler_features, 'scaler_features.pkl')
joblib.dump(scaler_labels, 'scaler_labels.pkl')

# 绘制训练和验证的损失曲线
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练和验证的MAE曲线
plt.plot(history.history['mae'], label='train_mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()
