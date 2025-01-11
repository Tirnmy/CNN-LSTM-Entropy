import os
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mymodels import *

device = torch.device("cuda:0")


class CountDataset(Dataset):
    def __init__(self, data_dir, label_path):
        self.data_dir = data_dir
        self.label_path = label_path
        self.df_label = pd.read_csv(self.label_path)
        f_lst = os.listdir(data_dir)
        self.id_lst = [int(f.rstrip('.csv')) for f in f_lst]
        self.fpath_lst = [os.path.join(data_dir, file) for file in f_lst]

    def __len__(self):
        return len(self.id_lst)

    def __getitem__(self, idx):
        fpath = self.fpath_lst[idx]
        df_data = pd.read_csv(fpath)
        np_array = df_data.to_numpy()
        data = torch.from_numpy(np_array).float()
        label = self.df_label.loc[self.df_label['enrollment_id'] == self.id_lst[idx], 'truth'].values[0]
        return data, label


class EntropyDataset(Dataset):
    def __init__(self, data_dir, label_path, e_weekend_dir, e_workday_dir):
        self.data_dir = data_dir  # count目录路径
        self.label_path = label_path
        self.e_weekend_dir = e_weekend_dir  # entropy-weekend 目录路径
        self.e_workday_dir = e_workday_dir  # entropy-workday 目录路径
        self.df_label = pd.read_csv(self.label_path)
        f_lst = os.listdir(data_dir)
        self.id_lst = [int(f.rstrip('.csv')) for f in f_lst]
        self.fpath_lst = [os.path.join(data_dir, f) for f in f_lst]  # count下文件相对路径列表
        self.wknd_lst = [os.path.join(e_weekend_dir, f) for f in os.listdir(e_weekend_dir)]  # entropy-weekend下文件相对路径列表
        self.wkdy_lst = [os.path.join(e_workday_dir, f) for f in os.listdir(e_workday_dir)]  # entropy-workday下文件相对路径列表

    def __len__(self):
        return len(self.id_lst)

    def __getitem__(self, idx):
        fpath = self.fpath_lst[idx]
        df_data = pd.read_csv(fpath)

        wknd_path = self.wknd_lst[idx]
        df_wknd = pd.read_csv(wknd_path)
        wkdy_path = self.wkdy_lst[idx]
        df_wkdy = pd.read_csv(wkdy_path)

        data_np_array = df_data.to_numpy()
        data = torch.from_numpy(data_np_array).float()
        wknd_np_array = df_wknd.to_numpy()
        wknd = torch.from_numpy(wknd_np_array).float()
        wkdy_np_array = df_wkdy.to_numpy()
        wkdy = torch.from_numpy(wkdy_np_array).float()
        label = self.df_label.loc[self.df_label['enrollment_id'] == self.id_lst[idx], 'truth'].values[0]
        return data, wknd, wkdy, label





if __name__ == '__main__':
    model = CLSA().to(device)

    # 添加 tensorboard
    name = "CLSA"
    writer_summary_path = os.path.join('./logs', name)
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_dir = os.path.join(writer_summary_path, current_time)
    writer = SummaryWriter(log_dir=log_dir, comment=name)

    train_data = CountDataset(data_dir='./data/train/count', label_path='./data/train/truth/train_truth.csv')
    test_data = CountDataset(data_dir='./data/test/count', label_path='./data/test/truth/test_truth.csv')

    # length 长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练数据集的长度：{}".format(train_data_size))
    print("测试数据集的长度：{}".format(test_data_size))

    # 利用 Dataloader 来加载数据集
    train_dataloader = DataLoader(train_data, batch_size=128)
    test_dataloader = DataLoader(test_data, batch_size=128)

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    # 学习率
    learning = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning)

    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0

    # 训练的轮次
    epoch = 30

    start_time = time.time()

    for i in range(epoch):
        print("-----第 {} 轮训练开始-----".format(i + 1))

        # 训练步骤开始
        model.train()  # 当网络中有dropout层、batchnorm层时能起作用
        for data in train_dataloader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # 优化器对模型调优
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print(end_time - start_time)  # 运行训练一百次后的时间间隔
                print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 测试步骤开始
        model.eval()  # dropout层、batchnormal，这些层不能起作用
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:  # 测试数据集提取数据
                inputs, targets = data  # 数据放到cuda上
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)  # 仅data数据在网络模型上的损失
                total_test_loss = total_test_loss + loss.item()  # 所有loss
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

        print("整体测试集上的Loss：{}".format(total_test_loss))
        print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
        total_test_step = total_test_step + 1

        model_dir = f"./model{name}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model, f"./{model_dir}/model_{i}.pth")  # 保存每一轮训练后的结果
        # torch.save(model.state_dict(),f"{model_dir}/model_{i}.pth") # 保存方式二
        print("模型已保存")

    writer.close()
