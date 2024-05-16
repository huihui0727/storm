import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

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

# 加载模型
print("加载模型")
model_path = "image_prediction_model.h5"
image_prediction_model = models.load_model(model_path)

# 加载测试数据
storm_id = input("请输入要读取的风暴ID: ")
storm_features, test_images, _ = load_single_dataset(storm_id, "test")

# 预测未来图像
future_images = image_prediction_model.predict(test_images)

# 显示三张预测结果
for i, image in enumerate(future_images[:3]):
    plt.imshow(image, cmap='gray')
    plt.title(f"Image {i+1} Prediction")
    plt.axis('off')
    plt.show()
