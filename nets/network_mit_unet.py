import torch
import torch.nn as nn
from models.mit_unet.mit_unet import MiT_Unet


class Net(nn.Module):
    def __init__(self, phi="b0"):
        super(Net, self).__init__()
        self.mit_unet = MiT_Unet(phi=phi)

    def forward(self, x):
        logist = self.mit_unet(x)
        return logist
