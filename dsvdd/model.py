import torch.nn as nn
import torch

class Deep_SVDD(nn.Module):
    def __init__(self, z_dim=9):
        super(Deep_SVDD, self).__init__()
        self.input_dim = 224 # 산과들에
        # self.input_dim = 200 # indian pines
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)
        self.leakly_relu = nn.LeakyReLU()


        self.conv1 = nn.Conv2d(self.input_dim, self.input_dim * 3, 5, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_dim * 3, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(self.input_dim * 3, self.input_dim * 2, 5, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(self.input_dim * 2, eps=1e-04, affine=False)
        # self.fc1 = nn.Linear(self.input_dim * 2 * 3 * 3, self.input_dim * self.z_dim, bias=False)
        self.fc1 = nn.Linear(self.input_dim * 2 * 5 * 5, self.input_dim * self.z_dim, bias=False) # test

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakly_relu(self.bn1(x))
        x = self.conv2(x)
        x = self.pool(self.leakly_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class C_AutoEncoder(nn.Module):
    def __init__(self, z_dim=18):
        super(C_AutoEncoder, self).__init__()
        self.z_dim = z_dim
        self.input_dim = 224 # 산과들에
        # self.input_dim = 200 # indian pines
        self.weight_channel = self.input_dim * 2
        self.weight_channel2 = self.input_dim * 3
        self.feature_maps = []

        self.leakly_relu = nn.LeakyReLU()

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

        self.concat_layer = nn.Conv2d(self.weight_channel * 2, self.weight_channel, 1)
        self.concat_layer2 = nn.Conv2d(self.weight_channel2 * 2, self.weight_channel2, 1)

        self.conv1 = nn.Conv2d(self.input_dim, self.input_dim * 3, 5, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_dim * 3, eps=1e-04, affine=False)
        # self.conv2 = nn.Conv2d(self.input_dim * 3, self.input_dim * 2, 5, bias=False, padding=1)
        self.conv2 = nn.Conv2d(self.input_dim * 3, self.input_dim * 2, 5, bias=False, padding=1) # test
        self.bn2 = nn.BatchNorm2d(self.input_dim * 2, eps=1e-04, affine=False)
        self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(self.input_dim * 2 * 3 * 3, self.input_dim * z_dim, bias=False) 
        self.fc1 = nn.Linear(self.input_dim * 2 * 5 * 5, self.input_dim * z_dim, bias=False) # test

        # self.deconv1 = nn.ConvTranspose2d(self.input_dim, self.input_dim * 2, 5, bias=False, padding=2, stride=3)
        self.deconv1 = nn.ConvTranspose2d(self.input_dim, self.input_dim * 2, 5, bias=False, padding=3, stride=3) # test
        self.bn3 = nn.BatchNorm2d(self.input_dim * 2, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(self.input_dim * 2, self.input_dim * 3, 5, bias=False, padding=1)
        self.bn4 = nn.BatchNorm2d(self.input_dim * 3, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(self.input_dim * 3, self.input_dim, 5, bias=False, padding=0)
        
    def encoder(self, x):
        self.x1 = self.conv1(x) # 100 224*3 9 9
        self.x1 = self.leakly_relu(self.bn1(self.x1))
        self.x2 = self.conv2(self.x1) # 100 224*2 7 7
        self.x2 = self.leakly_relu(self.bn2(self.x2))
        x3 = self.pool(self.x2)
        x4 = x3.view(x.size(0), -1) # 100 224*2*3*3
        return self.fc1(x4) # 100 224*18
   
    def decoder(self, x):
        # x = x.view(x.size(0), self.input_dim, 3, 3) # 100 224 3 3 
        x = x.view(x.size(0), self.input_dim, 5, 5)     # test
        x = self.deconv1(self.leakly_relu(x)) # 100 224*2 7 7
        # x = (x + self.x2) # add
        # x = torch.cat([x, self.x2], dim = 1) # concat
        # x = self.concat_layer(x)
        m = self.weight1(torch.cat([self.x2, x], dim = 1)) # asc
        x = (1 - m) * x + m * self.x2 # asc
        x = self.deconv2(self.leakly_relu(self.bn3(x))) # 100 224*3 9 9
        # x = (x + self.x1) # add
        # x = torch.cat([x, self.x1], dim = 1) # concat
        # x = self.concat_layer2(x)
        m = self.weight2(torch.cat([self.x1, x], dim = 1)) # asc
        x = (1 - m) * x + m * self.x1 # asc
        x = self.deconv3(self.leakly_relu(self.bn4(x))) # 100 224 13 13    
        return torch.sigmoid(x)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    @staticmethod
    def loss_func(x, x_hat):
        output = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
        return output