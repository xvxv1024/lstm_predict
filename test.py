import torch
from Model import LSTMModel
import torch.nn as nn
import dataloader
import math
from baseUtils import RMSELoss

device = 0

model = F107Model().to(device)
model.load_state_dict(torch.load(r'./checkpoints/model_weights_epoch_100.pt'))
with torch.no_grad():
    batch_id = 0
    for inputs, labels in val_dataloader:
        batch_id = batch_id + 1

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        criterion1 = nn.L1Loss()
        criterion2 = RMSELoss()

        loss_mae = criterion1(outputs.float(), labels.float())
        loss_rmse = criterion2(outputs.float(), labels.float())

        epoch_loss_mae += loss_mae
        epoch_loss_rmse += loss_rmse

mae = epoch_loss_mae / batch_id
rmse = epoch_loss_rmse / batch_id

print('Validation MAE: %.3f, Validation RMSE: %.3f' % (mae, rmse))
