import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, eps=1e-6):
        super().__init__()
        self.margin = margin
        self.eps = eps
    
    def forward(self, x1, x2, label):
        # Normalize embeddings
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)

        euclidean_distance = F.pairwise_distance(x1, x2, eps=self.eps)
        loss = (
            label * euclidean_distance.pow(2) 
            + (1 - label) * torch.clamp(self.margin - euclidean_distance, min=0).pow(2)
            )
        return loss.mean()