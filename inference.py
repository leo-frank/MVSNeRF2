from train import MVSVolume
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import cv2
from dtu_data import DTU_dataset
import copy
from pytorch_lightning.loggers import TensorBoardLogger

dataset = DTU_dataset(data_dir="./dtu_train/", mode='test', nviews=3, ndepths=256)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=True)
# print(type(dataset[0]))
model = MVSVolume.load_from_checkpoint('./tb_logs/default/version_17/checkpoints/epoch=1-step=54193.ckpt')
tensorboard = TensorBoardLogger(save_dir = "./tb_test_logs/")
trainer = pl.Trainer(logger=tensorboard, max_epochs=2, gpus=1)
trainer.predict(model, dataloaders=test_loader)

# # disable randomness, dropout, etc...
# model.eval()
# # predict with the model
# with torch.no_grad():
#     y_hat = model(x)