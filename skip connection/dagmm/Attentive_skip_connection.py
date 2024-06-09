import torch
import torch.nn as nn
import torch.nn.functional as F

class ASC(nn.Module):
    def __init__(self, channel=200):
        super().__init__()
        self.channel = channel
        self.weight = nn.Sequential(
            nn.Conv2d(self.channel * 2, self.channel, 1),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel, self.channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, e, d):
        m = self.weight(torch.cat([e, d], dim=1))
        output = (1 - m) * e + m * d
        return output