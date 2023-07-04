import numpy as np
import baseUtils
from torch.utils.data import Dataset, DataLoader
from baseUtils import MyDataset

# TODO 输入序列长度 输出序列长度 滑动窗口步长
input_length, output_length, step = 18, 9, 1

# TODO 使用第几个分量
i = 1
list_flux = []
with open('./fluxtable.txt', 'r') as f:
    line_num = 0
    for line in f:
        line_num = line_num + 1

        if line_num <= 19750:
            continue

        fields = line.split()
        obsflux = float(fields[3 + i])
        list_flux.append(obsflux)

# TODO 读取为 array
array_flux = np.array(list_flux).reshape(-1, 1)

# 求数据的均值和方差
mea = array_flux[:, 0].mean() # 94.62
sig = array_flux[:, 0].std()  # 33.53

# 标准化数据
array_flux[:, 0] = (array_flux[:, 0] - mea) / sig

# TODO 滑动窗口提取子序列
data = baseUtils.get_data(array_flux, input_length, output_length, step)
print(data.shape)  # torch.Size([20241, 27, 1])

batch_size = 32


def get_dataloader_train():
    dataset_train = MyDataset(data[:300])
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    return dataloader_train


def get_dataloader_valid():
    dataset_valid = MyDataset(data[300:400])
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    return dataloader_valid


def get_dataloader_test():
    dataset_test = MyDataset(data[-200:])
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    return dataloader_test
