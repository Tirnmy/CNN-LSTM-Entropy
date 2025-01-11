import torch
from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    2维卷积
    """
    def __init__(self, input_channels=1, num_classes=2):
        super(SimpleCNN, self).__init__()
        # 使用nn.Sequential定义模型
        self.model = nn.Sequential(
            # 7个卷积层
            nn.Conv2d(in_channels=input_channels, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2,1), padding=1),
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=1),
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2,1), padding=1),
            # (10,10)
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # torch.Size([1, 7, 10, 10])


            # 展平层
            nn.Flatten(),
            # 全连接层
            nn.Linear(7*10*10, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # 二分类
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 直接将输入传递给定义好的Sequential模型
        x = torch.unsqueeze(x, dim=1)
        x = self.model(x)
        return x

class TimeSeriesCNN(nn.Module):
    """
    1维卷积
    """
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


class CNNEntropy(nn.Module):
    def __init__(self, input_channels=7, num_classes=2):
        super(CNNEntropy, self).__init__()
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
            nn.ReLU()
        )

        self.model2 = nn.Sequential(
            nn.Linear(16 + 7 + 7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x, e1, e2):
        # 直接将输入传递给定义好的Sequential模型
        x = torch.reshape(x, (x.size(0), 7, 70))
        x = self.model(x)
        e1 = torch.flatten(e1, start_dim=1)
        e2 = torch.flatten(e2, start_dim=1)
        combined = torch.cat((x, e1, e2), dim=1)
        x = self.model2(combined)
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

class CNNLSTM(nn.Module):
    def __init__(self, input_channels=7, num_classes=2):
        super(CNNLSTM, self).__init__()
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

class CNNLSTMEntropy(nn.Module):
    def __init__(self, input_channels=7, num_classes=2):
        super(CNNLSTMEntropy, self).__init__()
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
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            # 16
        )
        self.model3 = nn.Sequential(
            nn.Linear(16 + 7 + 7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x, e1, e2):
        # 直接将输入传递给定义好的Sequential模型
        x = torch.reshape(x, (x.size(0), 7, 70))
        x = self.model(x)
        x = x.transpose(1, 2)
        x = self.model2(x)
        e1 = torch.flatten(e1, start_dim=1)
        e2 = torch.flatten(e2, start_dim=1)
        combined = torch.cat((x, e1, e2), dim=1)
        x = self.model3(combined)
        return x

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        # 定义注意力权重
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.xavier_uniform_(self.attention_weights.data)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size*2)
        # attention_weights: (hidden_size*2, 1)
        # 计算注意力得分
        attention_scores = torch.bmm(lstm_output, self.attention_weights.unsqueeze(0).expand(lstm_output.size(0), -1, -1))
        # attention_scores: (batch, seq_len, 1)
        attention_scores = F.softmax(attention_scores, dim=1)
        # 计算加权输出
        weighted_output = lstm_output * attention_scores
        # weighted_output: (batch, seq_len, hidden_size*2)
        # 求和得到最终的输出
        attention_output = weighted_output.sum(dim=1)
        # attention_output: (batch, hidden_size*2)
        return attention_output, attention_scores


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim = 2):
        super(BiLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        # 定义Bi-LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        # 定义注意力层
        self.attention = AttentionLayer(hidden_size)
        # 定义一个线性层用于输出
        self.fc = nn.Linear(hidden_size * 2, output_dim)

    def forward(self, x):
        # x 的形状是 (batch, seq_len, input_size)
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(1*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1*2, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM
        lstm_output, _ = self.lstm(x, (h0, c0))
        # torch.Size([1, 14, 128])

        # 将LSTM的输出传递给注意力层
        attention_output, attention_weights = self.attention(lstm_output)

        # 将注意力层的输出传递给全连接层
        output = self.fc(attention_output)

        # return output, attention_weights
        return output



class CLSA(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(CLSA, self).__init__()
        # 使用nn.Sequential定义模型
        self.model = nn.Sequential(
            # 7个卷积层
            nn.Conv2d(in_channels=input_channels, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=1),
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=1),
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=1),
            # (10,10)
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Conv2d(in_channels=7, out_channels=7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
            # torch.Size([1, 7, 14, 14])

        )

        self.model2 = nn.Sequential(
            # LSTM
            BiLSTMWithAttention(7*14, 64),

            # # 展平层，准备输入到全连接层
            # nn.Flatten(),
            # # 全连接层，输入特征数需要根据展平后的维度计算
            # nn.Linear(14*128, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # # 二分类
            # nn.Linear(128, num_classes)
        )


    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # 增加conv2d所需通道维度
        x = self.model(x)
        x = x.transpose(1, 2).flatten(2)
        x = self.model2(x)
        return x


if __name__ == '__main__':
    model = CLSA()
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
