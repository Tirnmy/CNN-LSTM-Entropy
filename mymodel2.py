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
        x = torch.transpose(x, 1, 2)
        x = self.model(x)
        x = self.model2(x)
        return x


class SEblock(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上


class CNN1dAttn(nn.Module):
    """
    CNN1dAttn
    """
    def __init__(self, input_channels=7, num_classes=2):
        super(CNN1dAttn, self).__init__()
        self.se_block = SEblock(ch_in=7)
        # 使用nn.Sequential定义模型
        self.model1 = nn.Sequential(
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
        x = torch.transpose(x, 1, 2)
        x = torch.unsqueeze(x, 2)
        # torch.Size([1, 7, 1, 70])
        x = self.se_block(x)
        # torch.Size([1, 7, 1, 70])
        x = torch.squeeze(x, 2)
        x = self.model1(x)
        # torch.Size([1, 16, 10])
        x = self.model2(x)
        return x


class BiLSTMModel(nn.Module):
    def __init__(self, input_size = 16, hidden_size = 32, num_layers = 1):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # 最后一个时间步的输出
        out = out[:, -1, :]
        return out


class CNN1dBiLSTM(nn.Module):
    def __init__(self, input_channels=7, num_classes=2):
        super(CNN1dBiLSTM, self).__init__()
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
            # LSTM
            BiLSTMModel(16, 32, 1),
            # torch.Size([1, 64])
            # 全连接层，输入特征数需要根据展平后的维度计算
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
    model = CNN1dAttn()
    print(model)
    x = torch.randn(1, 70, 7)
    print('x1', x)
    print(x.shape)
    outputs = model(x)


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