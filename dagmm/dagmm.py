import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

from Attentive_skip_connection import ASC

class DAGMM(nn.Module):
    def __init__(self, hyp):
        super(DAGMM, self).__init__()
        self.hyp = hyp
        self.weight_channel = self.hyp['hidden3_dim']
        self.weight_channel2 = self.hyp['hidden2_dim']
        self.weight_channel3 = self.hyp['hidden1_dim']
        self.e_layers1 = nn.Sequential(
            nn.Linear(hyp['input_dim'],hyp['hidden1_dim']),
            nn.Tanh()
        )
        self.e_layers2 = nn.Sequential(
            nn.Linear(hyp['hidden1_dim'],hyp['hidden2_dim']),
            nn.Tanh()
        )
        self.e_layers3 = nn.Sequential(
            nn.Linear(hyp['hidden2_dim'],hyp['hidden3_dim']),
            nn.Tanh()
        )
        self.e_layers4 = nn.Sequential(
            nn.Linear(hyp['hidden3_dim'],hyp['zc_dim'])
        )

        self.d_layers1 = nn.Sequential(
            nn.Linear(hyp['zc_dim'], hyp['hidden3_dim']),
            nn.Tanh()
        )
        self.d_layers2 = nn.Sequential(
            nn.Linear(hyp['hidden3_dim'],hyp['hidden2_dim']),
            nn.Tanh()
        )
        self.d_layers3 = nn.Sequential(
            nn.Linear(hyp['hidden2_dim'],hyp['hidden1_dim']),
            nn.Tanh()
        )
        self.d_layers4 = nn.Sequential(
            nn.Linear(hyp['hidden1_dim'],hyp['input_dim'])
        )

        self.estimation = nn.Sequential(
            nn.Linear(hyp['zc_dim']+2,hyp['hidden3_dim']),
            nn.Tanh(),
            nn.Dropout(p=hyp['dropout']),
            nn.Linear(hyp['hidden3_dim'],hyp['n_gmm']),
            nn.Softmax(dim=1)
        )

        self.weight1 = nn.Sequential(
            nn.Conv2d(self.weight_channel * 2, self.weight_channel, 1),
            nn.LeakyReLU(),
            nn.Conv2d(self.weight_channel, self.weight_channel, 3, 1, 1),
            nn.Sigmoid()
        )
        self.weight2 = nn.Sequential(
            nn.Conv2d(self.weight_channel2 * 2, self.weight_channel2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(self.weight_channel2, self.weight_channel2, 3, 1, 1),
            nn.Sigmoid()
        )
        self.weight3 = nn.Sequential(
            nn.Conv2d(self.weight_channel3 * 2, self.weight_channel3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(self.weight_channel3, self.weight_channel3, 3, 1, 1),
            nn.Sigmoid()
        )

        self.concat_layer1 = nn.Conv2d(self.hyp['hidden3_dim'] * 2, self.hyp['hidden3_dim'], 1) 
        self.concat_layer2 = nn.Conv2d(self.hyp['hidden2_dim'] * 2, self.hyp['hidden2_dim'], 1)
        self.concat_layer3 = nn.Conv2d(self.hyp['hidden1_dim'] * 2, self.hyp['hidden1_dim'], 1)

        self.lambda1 = hyp['lambda1']
        self.lambda2 = hyp['lambda2']


    def forward(self, x):
        enc1 = self.e_layers1(x)
        enc2 = self.e_layers2(enc1)
        enc3 = self.e_layers3(enc2)
        enc4 = self.e_layers4(enc3) #shape: 100 1

        dec = self.d_layers1(enc4)
        # output = (dec + enc3) # add
        # output = torch.concat([dec, enc3], dim = 2) # concat # hsi
        # # output = torch.concat([dec, enc1], dim = 1) # ip
        # output = output.permute(2, 1, 0) # hsi
        # # output = output.unsqueeze(0).permute(2, 1, 0)
        # output = self.concat_layer1(output)
        # output = output.permute(2, 1, 0)
        # print(enc3.shape)
        # print(dec.shape)
        output = self.asc(enc3, dec, self.hyp['hidden3_dim']) # asc
        # dec = self.d_layers2(dec)
        dec = self.d_layers2(output)
        # output = (dec + enc2) # add
        # output = torch.concat([dec, enc2], dim = 2) # concat # hsi
        # # output = torch.concat([dec, enc1], dim = 1) # ip
        # output = output.permute(2, 1, 0) # hsi
        # # output = output.unsqueeze(0).permute(2, 1, 0)
        # output = self.concat_layer2(output)
        # output = output.permute(2, 1, 0)
        output = self.asc(enc2, dec, self.hyp['hidden2_dim']) # asc     
        # dec = self.d_layers3(dec)
        dec = self.d_layers3(output)
        # output = (dec + enc1) # add
        # output = torch.concat([dec, enc1], dim = 2) # concat # hsi
        # # output = torch.concat([dec, enc1], dim = 1) # ip
        # output = output.permute(2, 1, 0) # hsi
        # # output = output.unsqueeze(0).permute(2, 1, 0)
        # output = self.concat_layer3(output)
        # output = output.permute(2, 1, 0)
        output = self.asc(enc1, dec, self.hyp['hidden1_dim']) # asc
        # dec = self.d_layers4(dec) #shape: 81 224
        dec = self.d_layers4(output)
        # dec = dec.unsqueeze(1) # hsi asc


        # rec_cosine = F.cosine_similarity(x, dec, dim=2) #shape: 81 # hsi
        rec_cosine = F.cosine_similarity(x, dec, dim=1) # indian pines
        rec_euclidean = F.pairwise_distance(x, dec,p=2) #shape: 81



        # z = torch.cat([enc4.squeeze(-1), rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim = 1) #shape: 81 3 # indian pines
        z = torch.cat([enc4, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim = 1) # test indian pines

        # z = torch.cat([enc4.squeeze(1), rec_euclidean, rec_cosine], dim = 1) # hsi

        gamma = self.estimation(z) #shape: 81 2

        return enc4,dec,z,gamma
    
    def asc(self, enc, dec, c):
        # enc = enc.reshape(c, self.hyp['patch'], self.hyp['patch']) # hsi
        # dec = dec.reshape(self.hyp['patch'], self.hyp['patch'], c).permute(2, 0, 1) # hsi
        enc = enc.reshape(self.hyp['patch'], self.hyp['patch'], c).permute(2, 0, 1)
        dec = dec.reshape(self.hyp['patch'], self.hyp['patch'], c).permute(2, 0, 1)
        if c == self.hyp['hidden3_dim']:
            m = self.weight1(torch.cat([enc, dec], dim = 0))
        elif c == self.hyp['hidden2_dim']:
            m = self.weight2(torch.cat([enc, dec], dim = 0))
        else:
            m = self.weight3(torch.cat([enc, dec], dim = 0))
        output = (1 - m) * enc + m * dec
        # output = output.reshape(self.hyp['patch'], self.hyp['patch'], -1) # hsi
        # output = output.reshape(self.hyp['patch'] ** 2, -1)  # hsi
        output = output.reshape(self.hyp['batch_size'], -1) 
        return output  
    
    @staticmethod
    def reconstruct_error(x, x_hat):   
        e = torch.tensor(0.0)

        for i in range(x.shape[0]):
            e += torch.dist(x[i], x_hat[i])
        return e / x.shape[0]
    
    @staticmethod
    def get_gmm_param(gamma, z):
        # z = z.mean(dim = 1)
        # gamma = gamma.mean(dim = 1)

        # z : 81 x 3
        # gamma : 81 x 2

        N = gamma.shape[0]
        ceta = torch.sum(gamma, dim=0) / N  #shape: 2
        
        mean = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(-2), dim = 0)
        mean = mean / (torch.sum(gamma, dim = 0)).unsqueeze(-1) #shape: 2 3

        z_mean = z.unsqueeze(1) - mean.unsqueeze(0) #shape: 81 2 3
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mean.unsqueeze(-1) * z_mean.unsqueeze(-2), dim = 0) / (torch.sum(gamma, dim = 0)).unsqueeze(-1).unsqueeze(-1) #shape: 2 3 3

            
        return ceta, mean, cov
    
    @staticmethod
    def sample_energy(ceta, mean, cov, zi, n_gmm):
        e = torch.tensor(0.0)
        cov_eps = torch.eye(mean.shape[1]) * (1e-12)
#         cov_eps = cov_eps.to(device)
        for k in range(n_gmm):
            d_k = zi - mean[k].unsqueeze(1) # 2 1 3

            inv_cov = torch.inverse(cov[k] + cov_eps) # 3 3

            e_k = torch.exp(-0.5 * torch.chain_matmul(torch.t(d_k), inv_cov, d_k)) # 1 1
            e_k = e_k / torch.sqrt(torch.abs(torch.det(2 * math.pi * cov[k])))
            e_k = e_k * ceta[k]
            e += e_k.squeeze()
        return -torch.log(e) + 1e-6
    
    

    def loss_func(self, x, dec, gamma, z):
        n_gmm = gamma.shape[1]

        #1
        recon_error = self.reconstruct_error(x.unsqueeze(0), dec)
        #2
        ceta, mean, cov = self.get_gmm_param(gamma, z)
      
        #3
        # z = z.mean(dim = 1)
        e = torch.tensor(0.0)
        for i in range(z.shape[0]): # 81
            ei = self.sample_energy(ceta, mean, cov, z[i].unsqueeze(1), n_gmm)
            e += ei
        
        p = torch.tensor(0.0)
        for i in range(n_gmm):
            p_k = torch.sum(1 / torch.diagonal(cov[i], 0))
            p += p_k


        loss = recon_error + (self.lambda1 / z.shape[0]) * e + self.lambda2 * p
        
        return loss, recon_error, e/z.shape[0], p