import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from torchmetrics.functional import auroc
import spectral
import os
from scipy import io
import torch, gc
import time
from torchsummary import summary
from dataset import DAGMM_dataset, Indian_pines_dataset
from dagmm import DAGMM
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

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

def load_model(hyp, device):
    model = DAGMM(hyp)
    model = model.to(device)
    try:
        model.load_state_dict(torch.load('./models/indian_pines/test/19_dagmm.pth'))
        print('success to load model')
    except Exception as e:
        print('Failed to load model: %s' % (e))
        exit(1)
    
    return model

def compute_threshold(model, train_loader,len_data, device):
    energies = np.zeros(shape=(len_data))
    step = 0
    energy_interval = 50
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            enc,dec,z,gamma = model(x)
            z, gamma = z.cpu(), gamma.cpu()
            m_prob, m_mean, m_cov = model.get_gmm_param(gamma, z)
            # z = z.mean(dim = 1)
            # gamma = gamma.mean(dim = 1)

            if len_data < 50:
                energy_interval = 2
            
            for i in range(z.shape[0]):
                zi = z[i].unsqueeze(1)
                sample_energy = model.sample_energy(m_prob, m_mean, m_cov, zi, gamma.shape[1])

                energies[step] += sample_energy.detach().item()
                step += 1

                if step % energy_interval == 0:
                    print('Iteration: %d    sample energy: %.4f' % (step, sample_energy))

    threshold = np.percentile(energies, 80)
    print('threshold:%.4f' %(threshold))
    
    return threshold

def main(hyp):
    has_threshold = True
    device = torch.device("cuda:0")
    model = load_model(hyp, device)
    if hyp['data'] == 'hsi':
        data = load_data('../data/hsi', 'data.raw')
        label = load_data('../data/hsi', 'label.npy')
        train_dataset = DAGMM_dataset(data, label, 'train')
        train_loader = DataLoader(dataset = train_dataset, batch_size = hyp['batch_size'], shuffle = True, drop_last = True)
        test_dataset = DAGMM_dataset(data, label, 'train')
        test_loader = DataLoader(dataset = test_dataset, batch_size = hyp['batch_size'], shuffle = False, drop_last = True)
    else:
        data = io.loadmat(hyp['data_dir'])
        data = data['indian_pines_corrected']
        label = io.loadmat(hyp['gt_dir'])
        label = label['indian_pines_gt']
        train_dataset = Indian_pines_dataset(data, label, 'train', hyp['patch'], hyp['normal'], hyp['ratio'])
        train_loader = DataLoader(dataset = train_dataset, batch_size = hyp['batch_size'], shuffle = True, drop_last = True)    
        test_dataset = Indian_pines_dataset(data, label, 'test', hyp['patch'], hyp['normal'], hyp['ratio'])
        test_loader = DataLoader(dataset = test_dataset, batch_size = hyp['batch_size'], shuffle = False, drop_last = True)    

    len_data = hyp['batch_size'] * len(train_loader)

    if has_threshold == False:
        threshold = compute_threshold(model,train_loader,len_data, device)
    else:
        # threshold = 5.5281 # before attention skip connection
        # threshold = 6.2218 # after attention skip connection
        # threshold = 6.9506 # dagmm 
        # threshold = 7.0396
        threshold = -1.55
    
    print('threshold: ', threshold)

    # test_dataset = DAGMM_dataset(data, label, 'test')
    # test_loader = DataLoader(dataset = test_dataset, batch_size = hyp['batch_size'], shuffle = True, drop_last = True)
    #indian_pines_dataset = Indian_pines_dataset(hyp['data_dir'], hyp['gt_dir'], 'test', hyp['patch'])
    # train_dataset.mode="test"
    # train_loader = DataLoader(dataset = train_dataset, batch_size = hyp['batch_size'], shuffle = True, drop_last = False) 
    # len_data = len(test_dataset)
    len_data = len(test_dataset)

    scores = np.zeros(shape=(len_data, 2))
    density = []
    label = []
    step = 0
    acc = []
    model.eval()
    with torch.no_grad():
        # for x_batch, y_batch in test_loader: # total m번 돌아감
        #     for x, y in zip(x_batch, y_batch):
        #     # x_batch - m 169 200 
        #         # x - 169 200
        #         x = x.to(device)
        #         _,_,z,gamma = model(x.reshape(-1, hyp['input_dim']))
        #         z, gamma = z.cpu(), gamma.cpu()
        #         m_prob, m_mean, m_cov = model.get_gmm_param(gamma, z)
        #         se = 0
                
        #         for i in range(z.shape[0]): # 100
        #             zi = z[i].unsqueeze(1)
        #             sample_energy = model.sample_energy(m_prob, m_mean, m_cov, zi,gamma.shape[1])
        #             se = sample_energy.detach().cpu().item()

                    
                    
        #             scores[step] = [int(y[i]), int(se < threshold)]
        #             step += 1
        #             density.append(se)
        #         label.append(y)
        #     accuracy = accuracy_score(scores[:, 0], scores[:, 1])
        #     acc.append(accuracy)
        # total_acc = sum(acc) / len(test_loader)
        for x, y in test_loader: # total m번 돌아감
            # x_batch - m 169 200 
                # x - 169 200
            x = x.to(device)
            _,_,z,gamma = model(x)
            z, gamma = z.cpu(), gamma.cpu()
            m_prob, m_mean, m_cov = model.get_gmm_param(gamma, z)
            se = 0
                
            for i in range(z.shape[0]): # 100
                zi = z[i].unsqueeze(1)
                sample_energy = model.sample_energy(m_prob, m_mean, m_cov, zi,gamma.shape[1])
                se = sample_energy.detach().cpu().item()
                    
                scores[step] = [int(y[i]), int(se < threshold)]
                step += 1
                density.append(se)
            label.append(y)
        accuracy = accuracy_score(scores[:, 0], scores[:, 1])
        acc.append(accuracy)
        total_acc = sum(acc) / len(test_loader)

    label, density = torch.cat(label).numpy(), np.array(density)
    precision, recall, fscore, _ = precision_recall_fscore_support(scores[:, 0], scores[:, 1])
    aupr = average_precision_score(scores[:, 0], scores[:, 1])

    print(f'accuracy - {accuracy}')
    print(f'fscore - {fscore}')
    print(f'precision - {precision}')
    print(f'recall - {recall}')
    print(f'AUPR - {aupr}')
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(scores[:, 0], scores[:, 1])*100))
    print('AUROC score: {:.2f}'.format(auroc(torch.as_tensor(scores[:,0]), torch.as_tensor(scores[:,1]).int())*100))

    normal = density[np.where(label==1)[0]]
    anomal = density[np.where(label==0)[0]]

    normal_max = np.percentile(normal, 90)
    normal_min = np.percentile(normal, 10)
    anomal_max = np.percentile(anomal, 90)
    anomal_min = np.percentile(anomal, 10)

    scores_in = normal[np.where(normal_min < normal)]
    scores_in = normal[np.where(normal_max > normal)]
    scores_out = anomal[np.where(anomal_min < anomal)]
    scores_out = anomal[np.where(anomal < anomal_max)]

    in_ = pd.DataFrame(scores_in, columns=['Inlier'])
    out_ = pd.DataFrame(scores_out, columns=['Outlier'])
    # in_ = pd.DataFrame(normal, columns=['Inlier'])
    # out_ = pd.DataFrame(anomal, columns=['Outlier'])    
    # print('Accuracy: %.4f  Precision: %.4f  Recall: %.4f  F-score: %.4f' % (accuracy, precision, recall, fscore))
    fig, ax = plt.subplots()
    in_.plot.kde(ax=ax, legend=True, title='Outliers vs Inliers (DAGMM)')
    out_.plot.kde(ax=ax, legend=True)
    # plt.xlim(-0.05, 0.08)
    # plt.ylim(0, 1.5)
    ax.grid(axis='x')
    ax.grid(axis='y')
    plt.title('attentive')
    plt.show()


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

    main(hyp)