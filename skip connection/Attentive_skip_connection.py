import torch
import torch.nn as nn
import torch.nn.functional as F

class ASC(nn.Module):
    """ Attentive Skip Connection
    """

    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, e, d):
        m = self.weight(torch.cat([e, d], dim=1))
        output = (1 - m) * d + m * e
        return output