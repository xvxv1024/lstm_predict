import torch
import torch.nn as nn


class F107Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=4, output_size=9):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        # 初始化隐状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 通过LSTM层
        out, _ = self.lstm(x, (h0, c0))
        # print(out.shape)  # torch.Size([8, 18, 64])
        # print(out[:, -1, :].shape)  # torch.Size([8, 64])
        # 取最后一个时间步的输出并通过全连接层
        out = self.fc(out[:, -1, :])
        out = out.reshape(-1, self.output_size, 1)
        return out
