import torch
from torch import nn
import numpy as np

class DDCNN(nn.Module):
    def __init__(self, n_classes, n_bands):
        super().__init__()
        self.n_classes = n_classes
        self.n_bands = n_bands
        self.growth_rate = 32
        self.alpha = 4
        self.gamma = 32
        self.k1 = 16 + 6 * self.gamma
        self.k3 = self.k1 / 2 + 16 * self.gamma
        self.input_covlayer = nn.Conv2d(in_channels = self.n_bands, out_channels = 16, kernel_size = 3, stride = 1)

        self.first_dense_layer = []
        for i in range(6):
            self.first_dense_layer.append(Dense_block(True, i + 1).cuda())

        self.second_dense_layer = []
        for i in range(16):
            self.second_dense_layer.append(Dense_block(False, i + 1).cuda())


        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(self.k1),
            nn.ReLU(),
            nn.Conv2d(in_channels = self.k1, out_channels = self.k1 // 2, kernel_size = 1),
            nn.Dropout(p = 0.1),
            nn.AvgPool2d(kernel_size= 2, stride= 2)
        )

        self.classifi_layer = nn.Sequential(
            nn.BatchNorm2d(int(self.k3)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(616, self.n_classes)
        )

    def forward(self, x):
        feature_maps = []
        sec_feature_maps = []
        output = self.input_covlayer(x)
        feature_maps.append(output)

        for i in range(6):
            output = self.first_dense_layer[i](output)
            feature_maps.append(output)
            for j in range(len(feature_maps) - 1):
                output = torch.cat([output, feature_maps[j]],1)

        output = self.transition_layer(output)
        sec_feature_maps.append(output)

        for i in range(16):
            output = self.second_dense_layer[i](output)
            sec_feature_maps.append(output)
            for j in range(len(sec_feature_maps) - 1):
                output = torch.cat([output, sec_feature_maps[j]],1)         

        output = self.classifi_layer(output)
        
        return output


class Dense_block(nn.Module):
    def __init__(self, check, num):
        super().__init__()
        self.first_inner_block_num = 6
        self.second_inner_block_num = 16
        self.gamma = 32
        self.alpha = 4
        self.num = num
        self.k1 = 16 + 6 * self.gamma
        self.check = check
        self.first_inner_block = nn.Sequential(
            nn.BatchNorm2d(16 + (self.num - 1) * self.gamma),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16 + (self.num - 1) * self.gamma, out_channels = self.gamma * self.alpha, kernel_size = 1, stride = 1),
            nn.Dropout(p = 0.1),

            nn.BatchNorm2d(self.gamma * self.alpha),
            nn.ReLU(),
            nn.Conv2d(in_channels = self.gamma * self.alpha, out_channels = self.gamma, kernel_size = 3, stride = 1, padding = 'same'),
            nn.Dropout(p = 0.1)
        )
        self.second_inner_block = nn.Sequential(
            nn.BatchNorm2d(self.k1 // 2 + (self.num - 1) * self.gamma),
            nn.ReLU(),
            nn.Conv2d(in_channels = self.k1 // 2 + (self.num - 1) * self.gamma, out_channels = self.gamma * self.alpha, kernel_size = 1, stride = 1),
            nn.Dropout(p = 0.1),

            nn.BatchNorm2d(self.gamma * self.alpha),
            nn.ReLU(),
            nn.Conv2d(in_channels = self.gamma * self.alpha, out_channels = self.gamma, kernel_size = 3, stride = 1, padding = 'same'),
            nn.Dropout(p = 0.1)
        )

    def forward(self, input):
        if self.check:
            inner_block = self.first_inner_block
        else:
            inner_block = self.second_inner_block

        b = inner_block(input)
        return b
