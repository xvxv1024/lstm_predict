import torch
from F107Model import F107Model
import torch.nn as nn
import dataloader
import math
from baseUtils import RMSELoss

device = 0
input_size = 1
hidden_size = 16
# model = F107Model(input_size, hidden_size).to(device)
model = F107Model(input_size, hidden_size)


test_dataloader = dataloader.get_dataloader_test()

model.load_state_dict(torch.load(r'./checkpoints/model_weights_epoch_40.pt'))
epoch_loss_mae = 0.0
epoch_loss_rmse = 0.0
with torch.no_grad():
    batch_id = 0
    for inputs, labels in test_dataloader:
        batch_id = batch_id + 1
        print(inputs.shape)
        print(labels.shape)
        # inputs = inputs.to(device)
        # labels = labels.to(device)

        outputs = model(inputs)
        # 反标准化
        outputs = outputs[:, :, :] * 33.53 + 94.62
        labels = labels[:, :, :] * 33.53 + 94.62
        
        criterion1 = nn.L1Loss()
        criterion2 = RMSELoss()

        loss_mae = criterion1(outputs.float(), labels.float())
        loss_rmse = criterion2(outputs.float(), labels.float())

        epoch_loss_mae += loss_mae
        epoch_loss_rmse += loss_rmse

mae = epoch_loss_mae / batch_id
rmse = epoch_loss_rmse / batch_id

print('test MAE: %.3f, test RMSE: %.3f' % (mae, rmse))

# test MAE: 3.519, test RMSE: 9.865
