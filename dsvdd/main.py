import os
import numpy as np
import spectral
import torch
from dataset import DSVDD_dataset, Indian_pines_dataset
from train import TrainerDeepSVDD
from torch.utils.data import DataLoader
from test import eval
import matplotlib.pyplot as plt
import pandas as pd 
from scipy import io

class Args:

    num_epochs=30
    # num_epochs_ae=30 # Indian pines
    num_epochs_ae=30
    lr=1e-4
    weight_decay=0.5e-6
    weight_decay_ae=0.5e-3
    lr_ae=1e-4
    lr_milestones=[50]
    batch_size=50 # Indian pines
    # batch_size=15
    pretrain=True
    latent_dim=25
    normal_class=16
    train=True
    patch=17
    ratio=70
    data='hsi'
    
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

args = Args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloaders = {}
if args.data == 'hsi':
    data = load_data('../data/hsi', 'data.raw')
    label = load_data('../data/hsi', 'label.npy')
    train_data = DSVDD_dataset(data, label, 'train')
    test_data = DSVDD_dataset(data, label, 'test')
    dataloaders['train'] = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, drop_last = True)
    dataloaders['test'] = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, drop_last = False)
else:
    indian_data = io.loadmat('../data/Indian_pines/Indian_pines_corrected')
    indian_data = indian_data['indian_pines_corrected']
    indian_label = io.loadmat('../data/Indian_pines/Indian_pines_gt')
    indian_label = indian_label['indian_pines_gt']
    indian_pines_train = Indian_pines_dataset(indian_data, indian_label, 'train', args.patch, args.normal_class)
    indian_pines_test = Indian_pines_dataset(indian_data, indian_label, 'test', args.patch, args.normal_class)
    dataloaders['train'] = DataLoader(indian_pines_train, batch_size = args.batch_size, shuffle = True, drop_last = True)
    dataloaders['test'] = DataLoader(indian_pines_test, batch_size = args.batch_size, shuffle = False, drop_last = False)    

deep_SVDD = TrainerDeepSVDD(args, dataloaders, device)

# AE pretrain
# if args.pretrain:
#     deep_SVDD.pretrain()

deep_SVDD.train()

model_loss = deep_SVDD.loss

labels, scores = eval(deep_SVDD.net, deep_SVDD.c, dataloaders['test'], device)
scores_in = scores[np.where(labels==1)[0]]
scores_out = scores[np.where(labels==0)[0]]

scores_in_max = np.percentile(scores_in, 90)
scores_in_min = np.percentile(scores_in, 10)
scores_out_max = np.percentile(scores_out, 90)
scores_out_min = np.percentile(scores_out, 10)

scores_in = scores_in[np.where(scores_in_min < scores_in)]
scores_in = scores_in[np.where(scores_in_max > scores_in)]
scores_out = scores_out[np.where(scores_out_min < scores_out)]
scores_out = scores_out[np.where(scores_out < scores_out_max)]

in_ = pd.DataFrame(scores_in, columns=['Inlier'])
out_ = pd.DataFrame(scores_out, columns=['Outlier'])


fig, ax = plt.subplots(1, 2)
in_.plot.kde(ax=ax[0], legend=True, title='Outliers vs Inliers (Deep SVDD)')
out_.plot.kde(ax=ax[0], legend=True)
# plt.xlim(-0.05, 0.08)
# plt.ylim(0, 1.5)
ax[0].grid(axis='x')
ax[0].grid(axis='y')

plt.plot(model_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('model loss for each epoch')

plt.show()