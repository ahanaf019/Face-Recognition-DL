import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from pathlib import Path
import os
import random
import pandas as pd
from torch.utils.data import Dataset

from utils import *
from datasets import FaceBBoxDataset
from models import FaceBBoxModel
from losses import ContrastiveLoss
from trainer import FaceBBoxTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


BATCH_SIZE = 64
IMAGE_SIZE = 224
LEARNING_RATE = 1e-4
NUM_EPOCHS = 70


train_paths, train_bboxes = get_image_bbox_dict('train')
val_paths, val_bboxes = get_image_bbox_dict('val')
test_paths, test_bboxes = get_image_bbox_dict('test')



# image, bbox = next(iter(val_db))
# bbox = [int(x * 224) for x in bbox]
# x, y, w, h = bbox
# print(bbox)
# plt.subplot(1,2,1)
# plt.imshow(image)
# plt.subplot(1,2,2)
# plt.imshow(image[y:y + h, x:x + w])
# plt.show()



train_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomAutocontrast(),
    transforms.RandomGrayscale(),
    transforms.RandomInvert(p=0.1),
])

test_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_db = FaceBBoxDataset(train_paths, train_bboxes, transforms=train_data_transforms)
val_db = FaceBBoxDataset(val_paths, val_bboxes, transforms=test_data_transforms)


train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=os.cpu_count(), prefetch_factor=2)
val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), prefetch_factor=2)


model = FaceBBoxModel(hid_dim=512).to(device)
loss_fn = nn.SmoothL1Loss()
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


trainer = FaceBBoxTrainer(model, train_loader, val_loader, optim, loss_fn)
trainer.train_model(num_epochs=NUM_EPOCHS)