import os
import torch
import time
import spectral
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from dataset import DAGMM_dataset, Indian_pines_dataset
from dagmm import DAGMM
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchsummary import summary

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

def plot_loss_moment(losses,hyp):
    _, ax = plt.subplots(figsize=(16, 9), dpi=80)
    ax.plot(losses, 'blue', label='train', linewidth=1)
    ax.set_title('Loss change in training')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(hyp['img_dir'], 'loss_dagmm.png'))
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(hyp):
    if hyp['data'] == 'hsi':
        data = load_data(hyp['data_dir'], 'data.raw')
        label = load_data(hyp['data_dir'], 'label.npy')
        train_dataset = DAGMM_dataset(data, label, 'train')
        train_loader = DataLoader(dataset = train_dataset, batch_size = hyp['batch_size'], shuffle = True, drop_last = True)
    else:      
        data = io.loadmat(hyp['data_dir'])
        data = data['indian_pines_corrected']
        label = io.loadmat(hyp['gt_dir'])
        label = label['indian_pines_gt']
        train_dataset = Indian_pines_dataset(data, label, 'train', hyp['patch'], hyp['normal'], hyp['ratio'])
        train_loader = DataLoader(dataset = train_dataset, batch_size = hyp['batch_size'], shuffle = True, drop_last = False)

    model = DAGMM(hyp)
    model = model.to(device)


    optim = torch.optim.Adam(model.parameters(),hyp['lr'], amsgrad=True)
    scheduler = MultiStepLR(optim, [5, 8], 0.1)

    loss_total = 0
    recon_error_total = 0
    e_total = 0
    best_e = 9999
    count = 0
    loss_plot = []
    start_time = time.time()

    model.train()

    for epoch in range(hyp['epochs']):
        for i, (input_data_batch, _) in enumerate(train_loader):
            for input_data in input_data_batch:
                input_data = input_data.to(device)
                
                optim.zero_grad()

                _,dec,z,gamma = model(input_data.reshape(-1, hyp['input_dim']))
                input_data,dec,z,gamma = input_data.cpu(),dec.cpu(),z.cpu(),gamma.cpu()
                loss, recon_error, e, __ = model.loss_func(input_data, dec, gamma, z)
        
                loss_total += loss.item() / len(input_data_batch)
                recon_error_total += recon_error.item() / len(input_data_batch)
                e_total += e.item() / len(input_data_batch)


                loss.backward()
                optim.step()

        
            if (i+1) % hyp['print_iter'] == 0:
                elapsed = time.time() - start_time
                
                log = "Time {:.2f}, Epoch [{}/{}], Iter [{}/{}], lr {} ".format(elapsed, epoch+1, hyp['epochs'], i+1, len(train_loader), optim.param_groups[0]['lr'])
                
                log+= 'loss {:.4f}, recon_error {:.4f}, energy {:.4f} '.format(loss_total/hyp['print_iter'], recon_error/hyp['print_iter'], e_total/hyp['print_iter'])
                loss_plot.append(loss_total/hyp['print_iter'])
                print(log)
                loss_total = 0
                recon_error_total = 0
                e_total = 0

    #     if best_e > loss_plot[-1]:
    #             best_e = loss_plot[-1]
    #             count = 0
    #     else:
    #             count += 1

    #     if count > 5:
    #         torch.save(model.state_dict(),
    #             os.path.join(hyp['save_path'], '{}_dagmm.pth'.format(epoch)))    
    #         break             
    
    #     scheduler.step()
    # if len(train_dataset) / hyp['batch_size'] > 20:
    #      print_iter = hyp['print_iter']
    # else:
    #      print_iter = 1

    # break_point = False

    # for epoch in range(hyp['epochs']):
    #     for i, (input_data, _) in enumerate(train_loader):
    #         # if break_point == 2:
    #         #      break_point = 0
    #         #      break
    #         input_data = input_data.to(device)
            
    #         optim.zero_grad()

    #         _,dec,z,gamma = model(input_data)
    #         input_data,dec,z,gamma = input_data.cpu(),dec.cpu(),z.cpu(),gamma.cpu()
    #         loss, recon_error, e, __ = model.loss_func(input_data, dec, gamma, z)
    
    #         loss_total += loss.item() 
    #         recon_error_total += recon_error.item()
    #         e_total += e.item()


    #         loss.backward()
    #         optim.step()

        
    #         if (i+1) % print_iter == 0:
    #             elapsed = time.time() - start_time
                
    #             log = "Time {:.2f}, Epoch [{}/{}], Iter [{}/{}], lr {} ".format(elapsed, epoch+1, hyp['epochs'], i+1, len(train_loader), optim.param_groups[0]['lr'])
                
    #             log += 'loss {:.4f}, recon_error {:.4f}, energy {:.4f} '.format(loss_total/print_iter, recon_error/print_iter, e_total/print_iter)
    #             loss_plot.append(loss_total/print_iter)
    #             loss_total = 0
    #             recon_error_total = 0
    #             e_total = 0
    #             print(log)
            # print(i)
            # print(loss_total/print_iter)
            # print(count)

            # if best_e > (loss_total/print_iter):
            #         best_e = (loss_total/print_iter)
            #         count = 0
            # else:
            #         count += 1

            # if count > 5:
            #     torch.save(model.state_dict(),
            #         os.path.join(hyp['save_path'], '{}_dagmm.pth'.format(epoch)))    
            #     count = 0
            #     if epoch > 2:
            #         break_point += 1
    
            scheduler.step()
           
        if (epoch+1) % hyp['savestep_epoch'] == 0:
                torch.save(model.state_dict(),
                    os.path.join(hyp['save_path'], '{}_dagmm.pth'.format(epoch)))
                
                
    plot_loss_moment(loss_plot,hyp)

if __name__ == "__main__":
	hyp={
	#  'input_dim':224, # 산과들에
     'input_dim':200, # indian pines
	 'hidden1_dim':60,
     'hidden2_dim':30,
     'hidden3_dim':10,
    #  'patch':10, # hsi asc
     'patch':10, # indian_pines
	 'zc_dim':1,
	 'emb_dim':10,
	 'n_gmm':2, # 산과들에
    #  'n_gmm':2,
	 'dropout':0.5,
	 'lambda1':0.1,
	 'lambda2':0.005,
	 'lr' :0.001,
	 'batch_size':100, # indian_pines
    #  'batch_size':100, 
	 'epochs': 20,
	 'print_iter':10,
     'ratio' : 70,
     'normal' : 1,
	 'savestep_epoch': 2,
	 'save_path': './models/indian_pines/test/',
	#  'data_dir': '../data/hsi',
     'data_dir': '../data/Indian_pines/Indian_pines_corrected',
     'gt_dir': '../data/Indian_pines/Indian_pines_gt',
	#  'img_dir': './result/indian_pines/attentive3/16',
     'img_dir': './result/indian_pines/test/',
     'data': 'indian_pines'
	}

	if not os.path.isdir('./models/'):
		os.mkdir('./models/')
	if not os.path.isdir('./result/'):
		os.mkdir('./result/')
		
	train(hyp)