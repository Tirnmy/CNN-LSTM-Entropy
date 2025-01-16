import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from train import CountDataset
from mymodels import *

# 参数设置
device = torch.device("cuda:0")

# 加载数据集
test_data = CountDataset(data_dir='./data/test/count', label_path='./data/test/truth/test_truth.csv')
test_dataloader = DataLoader(test_data, batch_size=128)

# 加载模型
# f_lst = os.listdir('./modelTimeSeriesCNN')
# for f in f_lst:
model_dir = './tmp/modelCLSA'
f_lst = os.listdir(model_dir)
for f in f_lst:
    model = torch.load(os.path.join(model_dir, f))
    model = model.to(device)
    model.eval()

    # 初始化指标变量
    y_true = []
    y_scores = []
    y_pred = []

    # 测试模型
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(outputs[:, 1].cpu().numpy())  # 只取正类概率
            y_pred.extend(predicted.cpu().numpy())

        # print('y_true')
        # print(y_true)
        # print('-'*20)
        # print('y_scores')
        # print(y_scores)
        # print('-'*20)
        # print('y_pred')
        # print(y_pred)
        # print('-'*20)
        # 计算指标
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        y_pred = np.array(y_pred)

        auc = roc_auc_score(y_true, y_scores)
        f1 = f1_score(y_true, y_pred, average='binary')
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        # 打印指标
        print('-' * 20)
        print(f)
        print(f"AUC: {auc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
