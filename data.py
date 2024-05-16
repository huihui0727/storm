import os
import json
import csv

data_dir = 'data/'


# 读取数据并准备训练集和测试集
def load_storm_data(storm_id):
    storm_path = os.path.join(data_dir, storm_id)
    storm_features = []
    storm_labels = []
    for file_name in sorted(os.listdir(storm_path)):
        if file_name.endswith('_features.json'):
            # 读取特征文件
            features_path = os.path.join(storm_path, file_name)
            with open(features_path, 'r') as f:
                features = json.load(f)
            storm_features.append(features)
        elif file_name.endswith('_label.json'):
            # 读取标签文件
            label_path = os.path.join(storm_path, file_name)
            with open(label_path, 'r') as f:
                label = json.load(f)
            storm_labels.append(label)

    return storm_features, storm_labels


# 保存数据到CSV文件
def save_to_csv(csv_file, data, storm_id):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image_id", "Storm_id", "relative_time", "Ocean", "Wind_speed"])  # 修改列名
        for i in range(len(data[0])):
            image_id = f"{storm_id}_{i:03d}"  # 修正image_id的生成方式
            feature = data[0][i]
            label = data[1][i]
            writer.writerow(
                [image_id, storm_id, feature['relative_time'], feature['ocean'], label['wind_speed']])  # 修改特征和标签的提取

# 加载风暴数据
storm_id = input("请输入要读取的风暴ID: ")
storm_features, storm_labels = load_storm_data(storm_id)

if not storm_features:
    print("找不到该风暴ID对应的数据。")
else:
    print("风暴ID:", storm_id)
    print("特征数量:", len(storm_features))
    print("标签数量:", len(storm_labels))
    print("第一个标签:", storm_labels[0])

    # 计算训练集和测试集的分割点
    split_index = int(len(storm_features) * 2 / 3)

    # 分割数据为训练集和测试集
    train_data = (storm_features[:split_index], storm_labels[:split_index])
    test_data = (storm_features[split_index:], storm_labels[split_index:])

    # 保存训练集数据到CSV文件
    train_csv_file = f"{storm_id}_train.csv"
    save_to_csv(train_csv_file, train_data, storm_id)

    # 输出训练集的前五行和后五行
    print("\n以下是训练集的前五行:")
    print("Image_id,Storm_id,relative_time,Ocean,Wind_speed")
    for i in range(min(5, len(train_data[0]))):
        image_id = f"{storm_id}_{i:03d}"
        feature = train_data[0][i]
        label = train_data[1][i]
        print(
            f"{image_id},{storm_id},{feature['relative_time']},{feature['ocean']},{label['wind_speed']}")

    print("\n以下是训练集的后五行:")
    for i in range(max(0, len(train_data[0]) - 5), len(train_data[0])):
        image_id = f"{storm_id}_{i:03d}"
        feature = train_data[0][i]
        label = train_data[1][i]
        print(
            f"{image_id},{storm_id},{feature['relative_time']},{feature['ocean']},{label['wind_speed']}")

    print(f"\n训练集数据已保存到文件: {train_csv_file}")

    # 保存测试集数据到CSV文件
    test_csv_file = f"{storm_id}_test.csv"
    save_to_csv(test_csv_file, test_data, storm_id)

    # 输出测试集的前五行和后五行
    print("\n以下是测试集的前五行:")
    print("Image_id,Storm_id,relative_time,Ocean,Wind_speed")
    for i in range(min(5, len(test_data[0]))):
        image_id = f"{storm_id}_{split_index + i:03d}"
        feature = test_data[0][i]
        label = test_data[1][i]
        print(
            f"{image_id},{storm_id},{feature['relative_time']},{feature['ocean']},{label['wind_speed']}")

    print("\n以下是测试集的后五行:")
    for i in range(max(0, len(test_data[0]) - 5), len(test_data[0])):
        image_id = f"{storm_id}_{split_index + i:03d}"
        feature = test_data[0][i]
        label = test_data[1][i]
        print(
            f"{image_id},{storm_id},{feature['relative_time']},{feature['ocean']},{label['wind_speed']}")

    print(f"\n测试集数据已保存到文件: {test_csv_file}")
