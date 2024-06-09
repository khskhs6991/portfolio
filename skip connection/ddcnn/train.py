import os
import torch
from torch import nn
import numpy as np
from model import DDCNN
from dataset import DDCNN_dataset
from scipy import io
from torch.utils.data import DataLoader, Dataset
import random
from torchmetrics import CohenKappa
import spectral
import copy

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_data(data_path, name, calibration=False, calibration_rate=1.0, calibration_path=None) -> np.ndarray:
    file_name, ext = os.path.splitext(name)
    if ext == ".npy": # label data
        data = np.load(os.path.join(data_path, name))
    elif ext == ".raw": # HSI data
        full_path = os.path.join(data_path, file_name)
        data = np.array(spectral.io.envi.open(full_path + ".hdr", full_path + ".raw").load())
        if calibration:
            _calibration_path = calibration_path if calibration_path else data_path
            dark_data = np.array(spectral.io.envi.open(os.path.join(_calibration_path, "DARKREF.hdr"), os.path.join(_calibration_path, "DARKREF.raw")).load()).mean(0)
            white_data = np.array(spectral.io.envi.open(os.path.join(_calibration_path, "WHITEREF.hdr"), os.path.join(_calibration_path, "WHITEREF.raw")).load()).mean(0)
            
            # Min-max scaling            
            data = (((data-dark_data)/(white_data-dark_data))*4095.0)*calibration_rate
            data = np.array(np.clip(data, 0, 4095), dtype=np.float32)
    else:
        raise ValueError(f"Unkown file format: {ext}")
    return data

data = load_data('../data/hsi', 'data.raw')
label = load_data('../data/hsi', 'label.npy')

patch_size = 9
batch_size = 100
boundary = int(patch_size / 2)
img_shape = 145
epochs = 100
n_classes = 2
ip_c = data.shape[2]
device = torch.device('cuda:0')

train_dataset = DDCNN_dataset(data, label, 'train')
valid_dataset = DDCNN_dataset(data, label, 'valid')

dataloaders = {}
dataloaders['train'] = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
dataloaders['valid'] = DataLoader(valid_dataset, batch_size = batch_size, shuffle = True, drop_last = True)


model = DDCNN(n_classes, ip_c).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss().to(device)
# kappa = CohenKappa(num_classes = n_classes - 1).to(device)

def save_model(model_state, model_name, save_dir = "./trained_model"):
    os.makedirs(save_dir, exist_ok = True)
    torch.save(model_state, os.path.join(save_dir, model_name))

for epoch in range(epochs + 1):
    losses = {}
    kappa_l = {}
    
    acc = {}

    best_valid_loss = 9999
    early_stop, early_stop_max = 0, 5

    training_loss = 0
    training_kappa = 0
    training_acc = 0

    for _, batch in enumerate(dataloaders['train']):
        images = batch[0].to(device)
        classes = batch[1].to(device)
        optimizer.zero_grad()
        model.train()

        pred = model(images)
        pred, classes = pred.cpu(), classes.cpu()
        loss = criterion(pred, classes)        
        # kappa_loss = kappa(pred, classes)  

        loss.backward()
        optimizer.step()

        pred = torch.argmax(pred, dim = 1)
        sub_acc = (pred == classes)

        training_acc += sub_acc.sum() / batch_size
        training_loss += loss.item()
        # training_kappa += kappa_loss.item()

    losses['train'] = training_loss / len(dataloaders['train'])
    # kappa_l['train'] = training_kappa / len(dataloaders['train'])
    acc['train'] = training_acc / len(dataloaders['train'])

    if epoch % 10 == 0:
        model.eval()

        valid_acc = 0
        valid_loss = 0
        valid_kappa = 0

        for _, batch in enumerate(dataloaders['valid']):
            images = batch[0].to(device)
            classes = batch[1].to(device)

            pred = model(images)
            loss = criterion(pred, classes)        
            # kappa_loss = kappa(pred, classes)  

            pred = torch.argmax(pred, dim = 1)
            sub_acc = (pred == classes)
        
            valid_acc += sub_acc.sum() / batch_size
            valid_loss += loss.item()
            # valid_kappa += kappa_loss.item()

        losses['valid'] = valid_loss / len(dataloaders['valid'])
        # kappa_l['valid'] = valid_kappa / len(dataloaders['valid'])
        acc['valid'] = valid_acc / len(dataloaders['valid'])

        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            early_stop = 0

            print(f"{epoch}/{epochs} - train acc: {acc['train']}, train loss: {losses['train']}, val_acc: {acc['valid']}, val_loss: {losses['valid']}")
            # print(f"{epoch}/{epochs} - train kappa: {kappa_l['train']}, val_kappa: {kappa_l['valid']}")
        else:
            early_stop += 1
            print(f"{epoch}/{epochs} - train acc: {acc['train']}, train loss: {losses['train']}, val_acc: {acc['valid']}, val_loss: {losses['valid']}")
            # print(f"{epoch}/{epochs} - train kappa: {kappa_l['train']}, val_kappa: {kappa_l['valid']}")

        best_model = copy.deepcopy(model.state_dict())
        save_model(best_model, "model.pth")

    if early_stop >= early_stop_max:
        break



    