import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
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

# 加载指定的风暴数据
storm_id = input("Enter the storm ID to predict: ")
print(f"Loading data for storm: {storm_id}")
features, images, labels = load_data(storm_id)

# 数据标准化
print("Scaling data...")
scaler_features = joblib.load('scaler_features.pkl')
scaler_labels = joblib.load('scaler_labels.pkl')
scaled_features = scaler_features.transform(features)
scaled_labels = scaler_labels.transform(labels.reshape(-1, 1))

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
X_features, X_images, y = create_sliding_window_features(scaled_features, images, scaled_labels, window_size)

# 加载模型
print("Loading the model...")
custom_objects = {"MeanSquaredError": tf.keras.losses.MeanSquaredError}
model = tf.keras.models.load_model('wind_speed_model.h5', custom_objects=custom_objects)

# 预测未来的13个风速
print("Predicting future wind speeds...")
future_predictions = model.predict([X_features[-13:], X_images[-13:]])

# 反标准化预测结果
future_predictions = scaler_labels.inverse_transform(future_predictions)

print("Predicted future wind speeds:", future_predictions)
