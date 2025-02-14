import torch
import torch.nn as nn
import torchvision


class PersonIdentificationModel(nn.Module):
    def __init__(self, hid_dim=512, out_dim=128):
        super().__init__()
        self.cnn = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.DEFAULT)
        self.fc = nn.Sequential(
            nn.Linear(2048, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, out_dim)
        )
    
    def forward(self, x1, x2):
        out1 = self.cnn(x1)
        out2 = self.cnn(x2)
        return out1, out2

    def inference(self, x):
        return self.cnn(x)