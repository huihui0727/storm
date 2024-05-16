import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array

data_dir = 'data/'  # 定义风暴数据的目录路径
image_height = 366  # 图像高度
image_width = 366   # 图像宽度

# 加载数据
def load_single_dataset(storm_id, data_type):
    storm_path = os.path.join(data_dir, f"{storm_id}_{data_type}")
    storm_features = []
    storm_images = []
    storm_labels = []
    for file_name in sorted(os.listdir(storm_path)):
        if file_name.endswith('_features.json'):
            # 读取特征文件
            features_path = os.path.join(storm_path, file_name)
            with open(features_path, 'r') as f:
                features = json.load(f)
            ocean = int(features['ocean'])
            storm_features.append([features['relative_time'], ocean])
        elif file_name.endswith('.jpg'):
            # 读取图片文件
            image_path = os.path.join(storm_path, file_name)
            image = load_img(image_path, target_size=(image_height, image_width), color_mode='grayscale')
            image_array = img_to_array(image) / 255.0  # 归一化
            storm_images.append(image_array)
        elif file_name.endswith('_label.json'):
            # 读取标签文件
            label_path = os.path.join(storm_path, file_name)
            with open(label_path, 'r') as f:
                label = json.load(f)
            storm_labels.append(label['wind_speed'])

    return np.array(storm_features), np.array(storm_images), np.array(storm_labels)

# 加载多个数据集
def load_datasets(storm_ids, data_type):
    all_features, all_images, all_labels = [], [], []
    for storm_id in storm_ids:
        features, images, labels = load_single_dataset(storm_id, data_type)
        all_features.append(features)
        all_images.append(images)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_images), np.concatenate(all_labels)

#生成未来图像预测
def build_image_prediction_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(input_shape[0] * input_shape[1], activation='sigmoid'),
        layers.Reshape((input_shape[0], input_shape[1]))
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')  # 使用均方误差作为损失函数
    return model

# 训练模型
def train_model(train_features, train_images):
    model = build_image_prediction_model(train_images.shape[1:])
    model.fit(train_images, train_images, epochs=num_epochs, batch_size=batch_size, verbose=1)
    return model

# 加载训练数据
storm_ids = ["bkh", "gic"]  # 添加你的其他数据集
train_features, train_images, train_labels = load_datasets(storm_ids, "train")

# 定义模型参数
num_epochs = 10
batch_size = 32

# 训练模型
print("\n训练模型")
model = train_model(train_features, train_images)

# 保存模型
model_path = "image_prediction_model.h5"
model.save(model_path)
print("\n模型训练完成并已保存。")


