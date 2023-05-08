import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class MyDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        x = self.data_tensor[index][:18]
        y = self.data_tensor[index][18:]
        return x, y

    def __len__(self):
        return len(self.data_tensor)

def get_data(array_flux, gt, horizon, step):
    inp_norm = get_strided_data_clust(array_flux, gt, horizon, step)
    inp_norm = torch.from_numpy(inp_norm)
    return inp_norm


def get_strided_data_clust(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt
    for i in range(1 + (raw_data.shape[0] - gt_size - horizon) // step):
        inp_te.append(raw_data[i * step:i * step + gt_size + horizon])

    inp_te_np = np.stack(inp_te)

    return inp_te_np