import torch
from torch import nn
import torch.nn.functional as F


class CNN1d(nn.Module):
    def __init__(self, input_channels=7, num_classes=2):
        super(CNN1d, self).__init__()
        # 使用nn.Sequential定义模型
        self.model = nn.Sequential(
            # 第一层卷积，in_channels=7（特征数），out_channels=16（输出通道数）
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            # torch.Size([1, 16, 10])
        )
        self.model2 = nn.Sequential(
            # 展平层，准备输入到全连接层
            nn.Flatten(),
            # 全连接层，输入特征数需要根据展平后的维度计算
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # 输出层，进行二分类
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 直接将输入传递给定义好的Sequential模型
        x = torch.reshape(x, (x.size(0), 7, 70))
        x = self.model(x)
        x = self.model2(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size = 32, hidden_size = 64, num_layers = 1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # 最后一个时间步的输出
        out = out[:, -1, :]
        return out


class CNN1dLSTM(nn.Module):
    def __init__(self, input_channels=7, num_classes=2):
        super(CNN1dLSTM, self).__init__()
        # 使用nn.Sequential定义模型
        self.model = nn.Sequential(
            # 第一层卷积，in_channels=7（特征数），out_channels=16（输出通道数）
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.model2 = nn.Sequential(
            # LSTM
            LSTMModel(32, 64, 1),
            # 1*64
            # 展平层，准备输入到全连接层
            nn.Flatten(),
            # 全连接层，输入特征数需要根据展平后的维度计算
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            # 输出层，进行二分类
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # 直接将输入传递给定义好的Sequential模型
        x = torch.reshape(x, (x.size(0), 7, 70))
        x = self.model(x)
        # x = x.transpose(1, 2)
        # x = self.model2(x)
        return x

class CNN1dAttnLSTM:
    def __init__(self, input_channels=7, num_classes=2):
        super(CNN1dAttnLSTM, self).__init__()
        # 使用nn.Sequential定义模型
        self.model = nn.Sequential(
            # 第一层卷积，in_channels=7（特征数），out_channels=16（输出通道数）
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # 16*36
            # 第二层卷积，使用上一次池化后的输出通道数作为输入通道数
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # 32*19
        )
        self.model2 = nn.Sequential(
            # LSTM
            LSTMModel(32, 64, 1),
            # 1*64
            # 展平层，准备输入到全连接层
            nn.Flatten(),
            # 全连接层，输入特征数需要根据展平后的维度计算
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            # 输出层，进行二分类
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # 直接将输入传递给定义好的Sequential模型
        x = torch.reshape(x, (x.size(0), 7, 70))
        x = self.model(x)
        x = x.transpose(1, 2)
        x = self.model2(x)
        return x


if __name__ == '__main__':
    model = CNN1d()
    x = torch.randn(1, 70, 7)
    outputs = model(x)
    print('x', x)
    print(x.shape)

    # x = torch.randn(1, 70, 7)
    # e1 = torch.randn(1, 1, 7)
    # e2 = torch.randn(1, 1, 7)
    # outputs = model(x, e1, e2)
    # print('x', x)
    # print(x.shape)
    # print('e1', e1)
    # print(e1.shape)
    # print('e2', e2)
    # print(e2.shape)

    # outputs = x.transpose(0, 1).flatten(1)

    print('outputs', outputs)
    print(outputs.shape)