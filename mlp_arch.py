import torch
from torch import nn, optim
import torchmetrics
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Callback

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from lightning.pytorch.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl

from helpers import KYBER_Q
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GetGrad(Callback):
    def __init__(self):
        super().__init__()
        self.input_grads = []
#
    def on_train_epoch_end(self, trainer, pl_module):
        self.input_grads.append(pl_module.grad)

class PICallback(Callback):
    """PyTorch Lightning PI callback."""

    def __init__(self):
        super().__init__()
        self.PI = []

    def on_validation_end(self, trainer, pl_module):
        self.PI.append(trainer.logged_metrics["PI"].item())

    def on_train_epoch_end(self, trainer, pl_module):
        self.input_grads.append(pl_module.grad)
class EarlyStopping_(EarlyStopping, Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ModelCheckpoint_(ModelCheckpoint, Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
# class MLP_SEC(pl.LightningModule):
#
#     def __init__(self, in_dim, out_dim=5):
#         super().__init__()
#         self.in_dim = in_dim
#         self.layers = nn.Sequential(
#         nn.BatchNorm1d(self.in_dim),
#         nn.Linear(self.in_dim, 500),
#         nn.BatchNorm1d(500),
#         nn.Dropout(0.2),
#         nn.LeakyReLU(),
#         nn.Linear(500, 100),
#         nn.LeakyReLU(),
#         nn.Linear(100, out_dim),
#         nn.Softmax(dim=1)
#         )
#         self.automatic_optimization = False
#         self.nll = nn.NLLLoss()
#         # self.save_hyperparameters()
#     def forward(self, x):
#         return self.layers(x)
#     def training_step(self, batch, batch_idx):
#         opt = self.optimizers()
#         x, y = batch
#         x.requires_grad  = True
#         y_hat = self.layers(x)
#         y_hat = torch.add(y_hat, 1e-15)
#         y_hat = torch.log2(y_hat)
#         opt.zero_grad()
#         loss = self.nll(y_hat, y)
#         self.manual_backward(loss)
#         opt.step()
#         self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
#         # self.grad = torch.abs(x.grad)
#         # self.grad =  torch.mean(self.grad, 0).numpy(force=True)
#         # self.bnweights = self.layers[0].weight.detach().numpy()
#         return loss
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.layers(x)
#         y_hat = torch.add(y_hat, 1e-15)
#         y_hat = torch.log2(y_hat)
#         loss = self.nll(y_hat, y)
#         self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
#         self.log_dict({"PI":  ENT_S - loss})
#         return loss
#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         x, y = batch
#         y_hat = self.layers(x)
#         # y_hat = F.softmax(y_hat, dim=1)
#         return y_hat
#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-4)
#         return optimizer
# class MLP(pl.LightningModule):

    # def __init__(self, in_dim, out_dim=3329, get_grad=False):
    #     super().__init__()
    #     self.in_dim = in_dim
    #     self.layers = nn.Sequential(
    #     nn.BatchNorm1d(self.in_dim),
    #     nn.Linear(self.in_dim, 500),
    #     nn.BatchNorm1d(500),
    #     nn.Dropout(0.2),
    #     nn.LeakyReLU(),
    #     nn.Linear(500, 100),
    #     nn.LeakyReLU(),
    #     nn.Linear(100, out_dim),
    #     nn.Softmax(dim=1)
    #     )
    #     # self.lossfn = nn.CrossEntropyLoss()
    #     self.automatic_optimization = False
    #
    #     self.lossfn = nn.NLLLoss()
    #     if out_dim == KYBER_Q:
    #         self.ent = ent_ZQ
    #     elif out_dim == HW_Q:
    #         self.ent = ent_HWQ
    #
    #     # self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_dim)
    #     self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_dim)
    #     self.get_grad = get_grad
    # def forward(self, x):
    #     return self.layers(x)
    # def training_step(self, batch, batch_idx):
    #     opt = self.optimizers()
    #     x, y = batch
    #     x.requires_grad  = True
    #     y_hat = self.layers(x)
    #     y_hat = torch.add(y_hat, 1e-15)
    #     y_hat = torch.log2(y_hat)
    #     opt.zero_grad()
    #     loss = self.lossfn(y_hat, y)
    #     self.manual_backward(loss)
    #     opt.step()
    #     self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    #     if self.get_grad:
    #         #self.grad = self.layers[0].weight.detach().numpy(force=True)
    #         grad = torch.abs(x.grad)
    #         grad =  torch.mean(grad, 0).numpy(force=True)
    #         self.grad = grad
    #     return loss
    #
    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.layers(x)
    #     y_hat = torch.add(y_hat, 1e-15)
    #     y_hat = torch.log2(y_hat)
    #     loss = self.lossfn(y_hat, y)
    #     self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    #     self.log_dict({"PI":  self.ent - loss})
    #     # self.valid_acc.update(y_hat, y)
    #     # self.log('valid_acc', self.valid_acc, prog_bar=True, on_epoch=True)
    #     return loss
    # # def on_validation_epoch_end(self):
    # #     self.valid_acc.reset()
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     x, y = batch
    #     y_hat = self.layers(x)
    #     # y_hat = F.softmax(y_hat, dim=1)
    #     return y_hat
    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-4)
    #     return optimizer
class MLP_BIN(pl.LightningModule):

    def __init__(self, in_dim, out_dim=2):
        super().__init__()
        self.in_dim = in_dim
        self.layers = nn.Sequential(
        nn.BatchNorm1d(self.in_dim),
        nn.Linear(self.in_dim, 1000),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(1000, 100),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(100, out_dim),
        )
        # self.lossf = nn.BCEWithLogitsLoss()
        self.lossf = nn.NLLLoss()
    def forward(self, x):
        return self.layers(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(device)
        y = y.type(torch.LongTensor) # <---- Here (casting)
        y = y.to(device)
        y_trans = self.layers(x)
        y_hat =F.softmax(y_trans, dim=1)
        y_hat = torch.log2(y_hat)
        loss = self.lossf(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        # self.log_dict({"train_loss": loss})
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(device)
        y = y.type(torch.LongTensor) # <---- Here (casting)
        y = y.to(device)
        y_trans = self.layers(x)
        y_hat =F.softmax(y_trans, dim=1)
        y_hat = torch.log2(y_hat)
        loss = self.lossf(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = x.to(device)
        y = y.type(torch.LongTensor) # <---- Here (casting)
        y = y.to(device)
        y_trans = self.layers(x)
        y_hat =F.softmax(y_trans, dim=1)
        return y_trans, y
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
class MLP_BIN_BCE(pl.LightningModule):

    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.in_dim = in_dim
        self.layers = nn.Sequential(
        nn.BatchNorm1d(self.in_dim),
        nn.Linear(self.in_dim, 1000),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(1000, 100),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(100, out_dim),
        )
        self.lossf = nn.BCELoss()
        # self.lossf = nn.NLLLoss()
    def forward(self, x):
        return self.layers(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(device)
        y = y.type(torch.float) # <---- Here (casting)
        y = y.to(device)
        y_trans = self.layers(x)
        y_hat =F.sigmoid(y_trans)
        # y_hat = torch.log2(y_hat)
        loss = self.lossf(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        # self.log_dict({"train_loss": loss})
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(device)
        y = y.type(torch.float) # <---- Here (casting)
        y = y.to(device)
        y_trans = self.layers(x)
        y_hat =F.sigmoid(y_trans)
        # y_hat =F.softmax(y_trans, dim=1)
        # y_hat = torch.log2(y_hat)
        loss = self.lossf(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = x.to(device)
        y = y.type(torch.float) # <---- Here (casting)
        y = y.to(device)
        y_trans = self.layers(x)
        # y_hat =F.softmax(y_trans, dim=1)
        y_hat =F.sigmoid(y_trans)
        return y_trans, y, y_hat
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

class LeakageData(Dataset):
    def __init__(self, traces, labels, mode=None):
        self.traces = traces
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.traces[idx], self.labels[idx]
