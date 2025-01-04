import torch
from torch import nn


class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels=7, num_classes=2):
        super(TimeSeriesCNN, self).__init__()
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
            # 展平层，准备输入到全连接层
            nn.Flatten(),
            # 全连接层，输入特征数需要根据展平后的维度计算
            nn.Linear(32 * 19, 64),  # 根据池化层后的尺寸调整
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            # 输出层，假设我们要进行二分类
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        # 直接将输入传递给定义好的Sequential模型
        x = torch.reshape(x, (x.size(0), 7, 70))
        x = self.model(x)
        return x


class CNNAE(nn.Module):
    def __init__(self):
        super(CNNAE, self).__init__()
        # 使用nn.Sequential定义模型
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # 246
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # 124
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # 63
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 126
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 252


            nn.Flatten(),
            # 全连接层
            nn.Linear(252, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = TimeSeriesCNN()
    inputs = torch.randn(1, 70, 7)
    outputs = model(inputs)
    print('inputs', inputs)
    print(inputs.shape)
    print('outputs', outputs)
    print(outputs.shape)
