import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class PersonIdentificationModel(nn.Module):
    def __init__(self, hid_dim=512, out_dim=128):
        super().__init__()
        self.cnn = self.cnn = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1")
        self.cnn.classifier = nn.Sequential(
            nn.Linear(1536, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, out_dim)
        )
    
    def forward(self, x1, x2):
        out1 = F.normalize(self.cnn(x1), p=2, dim=1)
        out2 = F.normalize(self.cnn(x2), p=2, dim=1)
        return out1, out2

    def inference(self, x):
        return F.normalize(self.cnn(x), p=2, dim=1)


class FaceBBoxModel(nn.Module):
    def __init__(self, hid_dim=512):
        super().__init__()
        self.cnn = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.DEFAULT)
        self.cnn.fc = nn.Sequential(
            nn.Linear(2048, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(True),
            # outputs x1, y1, w, h
            nn.Linear(hid_dim, 4)
        )
    
    def forward(self, x):
        return self.cnn(x)