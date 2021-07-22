import torch
import torch.nn as nn
import torch.nn.functional as F

class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        self.layer_1 = nn.Linear(num_features, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, 104)

    def forward(self, x):
        x = F.tanh(self.layer_1(x))
        x = F.tanh(self.layer_2(x))
        x = F.tanh(self.layer_3(x))
        x = self.layer_out(x)
        return x 
