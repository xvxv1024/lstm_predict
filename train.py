import torch
from F107Model import F107Model
import torch.nn as nn
import dataloader
import math
from baseUtils import RMSELoss

device = 0
# 实例化模型
input_size = 1
hidden_size = 64
model = F107Model(input_size, hidden_size).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataloader = dataloader.get_dataloader_train()
val_dataloader = dataloader.get_dataloader_valid()
# 训练模型
for epoch in range(1, 301):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if epoch % 5 == 0:
        mae = 0.0
        mse = 0.0
        total = 0
        epoch_loss_mae = 0.0
        epoch_loss_rmse = 0.0
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

        print('Epoch: %d, Training Loss: %.3f, Validation MAE: %.3f, Validation RMSE: %.3f' %
              (epoch, running_loss / len(train_dataloader), mae, rmse))

        torch.save(model.state_dict(), './checkpoints/model_weights_epoch_{}.pt'.format(epoch))
