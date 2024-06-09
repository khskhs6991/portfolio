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

def load_model(n_classes, ip_c):
    model = DDCNN(n_classes, ip_c)
    model = model.to(device)
    try:
        model.load_state_dict(torch.load('./trained_model/model.pth'))
        print('success to load model')
    except Exception as e:
        print('Failed to load model: %s' % (e))
        exit(1)
    
    return model

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

test_acc = 0

dataloaders = {}

test_dataset = DDCNN_dataset(data, label, 'test')

dataloaders['test'] = DataLoader(test_dataset, batch_size = 100, shuffle = True, drop_last = True)
class_per_acc = [[] for _ in range(n_classes)]
# running_kappa = 0
output = []
ground_truth = []
acc = 0

model = load_model(n_classes, ip_c)
model.eval()

# kappa = CohenKappa(task = 'multiclass', num_classes = n_classes - 1).to(device)

with torch.no_grad():
    for _, batch in enumerate(dataloaders['test']):
        images = batch[0].to(device)
        classes = batch[1].to(device)

        pred = model(images)
    
        pred = torch.argmax(pred, dim = 1)
        # output.extend(torch.argmax(pred.cpu(), dim = 1).tolist())
        # ground_truth.extend(classes.cpu())
        sub_acc = (pred == classes)
        # kappa_loss = kappa(pred, classes)  
        acc += sub_acc.sum() / batch_size
        # running_kappa += kappa_loss.item()

        for i in range(len(pred)):
            if pred[i] == classes[i]:
                class_per_acc[classes[i]].append(1)
            else:
                class_per_acc[classes[i]].append(0)


    oa = acc / len(dataloaders['test']) * 100
    # oa = accuracy_score(output, ground_truth)
    # kappa_l = running_kappa / len(dataloaders['test'])
    print('----------------------')
    # print(gt[i])
    print(oa)
    # print(kappa_l)
    print('----------------------')
    for i in range(n_classes):
        print(f'{i} class : {sum(class_per_acc[i]) / len(class_per_acc[i]) * 100} %')