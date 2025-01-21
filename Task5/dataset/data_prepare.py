import pandas as pd
import shutil
import os
from sklearn.utils import shuffle as reset


# 数据集划分函数：将数据划分为训练集和测试集
def train_test_split(data, test_size=0.1, shuffle=True, random_state=304):
    """
    划分数据集为训练集和测试集
    
    参数:
    - data: 数据集
    - test_size: 测试集比例
    - shuffle: 是否随机打乱数据
    - random_state: 随机种子，保证结果可复现
    
    返回:
    - train: 训练集
    - test: 测试集
    """
    if shuffle:
        data = reset(data, random_state=random_state)  # 随机打乱数据
    split_index = int(len(data) * test_size)  # 计算测试集的分割点
    train = data[split_index:].reset_index(drop=True)  # 训练集
    test = data[:split_index].reset_index(drop=True)  # 测试集
    return train, test


# 数据集准备函数：读取标签，划分训练集、验证集和测试集，复制文件
def dataset_prepare():
    """
    数据集准备函数，主要功能包括：
    1. 划分训练集和验证集
    2. 复制图像和文本文件到相应的文件夹
    3. 保存训练集和验证集的标签文件
    """
    # 读取测试集标签
    test_label = pd.read_csv('c:/Users/21768/Desktop/当代人工智能/Task5/dataset/test_without_label.txt', encoding='utf-8')
    test_num = test_label['guid'].values  # 获取测试集的ID
    
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs('./dataset/test/img', exist_ok=True)
    os.makedirs('./dataset/test/text', exist_ok=True)
    
    # 将测试集图像和文本文件复制到目标文件夹
    for num in test_num:
        img_path = f'./dataset/data/{num}.jpg'
        img_dest = f'./dataset/test/img/{num}.jpg'
        text_path = f'./dataset/data/{num}.txt'
        text_dest = f'./dataset/test/text/{num}.txt'
        shutil.copy(img_path, img_dest)  # 复制图像文件
        shutil.copy(text_path, text_dest)  # 复制文本文件

    # 读取训练集标签
    raw_label = pd.read_csv('c:/Users/21768/Desktop/当代人工智能/Task5/dataset/train.txt', encoding='utf-8')
    # 将训练集标签划分为训练集和验证集
    train_label, val_label = train_test_split(raw_label)

    # 对训练集和验证集按照guid排序
    train_label = train_label.sort_values(by="guid", ascending=True)
    val_label = val_label.sort_values(by="guid", ascending=True)
    
    # 将训练集和验证集标签保存为csv文件
    train_label.to_csv('./dataset/train/train.csv', index=False)
    val_label.to_csv('./dataset/val/val.csv', index=False)

    # 创建目标文件夹
    os.makedirs('./dataset/train/img', exist_ok=True)
    os.makedirs('./dataset/train/text', exist_ok=True)
    os.makedirs('./dataset/val/img', exist_ok=True)
    os.makedirs('./dataset/val/text', exist_ok=True)

    # 将训练集图像和文本文件复制到目标文件夹
    for num in train_label['guid'].values:
        img_path = f'./dataset/data/{num}.jpg'
        img_dest = f'./dataset/train/img/{num}.jpg'
        text_path = f'./dataset/data/{num}.txt'
        text_dest = f'./dataset/train/text/{num}.txt'
        shutil.copy(img_path, img_dest)
        shutil.copy(text_path, text_dest)

    # 将验证集图像和文本文件复制到目标文件夹
    for num in val_label['guid'].values:
        img_path = f'./dataset/data/{num}.jpg'
        img_dest = f'./dataset/val/img/{num}.jpg'
        text_path = f'./dataset/data/{num}.txt'
        text_dest = f'./dataset/val/text/{num}.txt'
        shutil.copy(img_path, img_dest)
        shutil.copy(text_path, text_dest)


# 调用数据集准备函数
if __name__ == '__main__':
    dataset_prepare()
