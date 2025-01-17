import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from train_entropy import EntropyDataset

# 参数设置
device = torch.device("cuda:0")

# 加载数据集
test_data = EntropyDataset(data_dir='./data/test/count', label_path='./data/test/truth/test_truth.csv',
                               e_weekend_dir='./data/test/week/entropy-weekend', e_workday_dir='./data/test/week/entropy-workday')
test_dataloader = DataLoader(test_data, batch_size=128)

# 加载模型
# f_lst = os.listdir('./modelTimeSeriesCNN')
# for f in f_lst:
# f = "./modelCLSAEntropy/model_6"
# model = CLSAEntropy()  # 导入网络结构
# model.load_state_dict(torch.load(f, weights_only=False))  # 导入网络的参数
# model = model.to(device)
# model.eval()

df = pd.DataFrame(index=range(20), columns=['AUC', 'F1-score', 'Precision', 'Recall', 'Accuracy'])
for i in range(20):
    model_dir = ''
    model = torch.load(os.path.join(model_dir, f'model_{i}.pth'))
    model = model.to(device)
    model.eval()

    # 初始化指标变量
    y_true = []
    y_scores = []
    y_pred = []

    # 测试模型
    with torch.no_grad():
        for inputs, wknd, wkdy, labels in test_dataloader:
            inputs, wknd, wkdy, labels = inputs.to(device), wknd.to(device), wkdy.to(device), labels.to(device)
            outputs = model(inputs, wknd, wkdy)
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
        print(f'model_{i}.pth')
        print(f"AUC: {auc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        df.loc[i, 'AUC'] = f"{auc:.4f}"
        df.loc[i, 'F1-score'] = f"{f1:.4f}"
        df.loc[i, 'Precision'] = f"{precision:.4f}"
        df.loc[i, 'Recall'] = f"{recall:.4f}"
        df.loc[i, 'Accuracy'] = f"{accuracy:.4f}"

print(df)