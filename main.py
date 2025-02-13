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
from utils import *
from datasets import FaceDetectionDataset
from models import FaceDetectionModel
from losses import ContrastiveLoss
from trainer import FaceDetectorTrainer


device = 'cuda' if torch.cuda.is_available() else 'cpu'


BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE = 1e-4
NUM_EPOCHS = 70

def get_person_image_dict(subset):
    persons = sorted(glob(f'{os.environ["HOME"]}/Datasets/CelebA/img_celeba/{subset}/*'))

    person_dict = {}
    for person in persons:
        person_images = sorted(glob(f'{person}/*'))
        person_dict[f'{Path(person).stem}'] = person_images
    return person_dict

train_dict = get_person_image_dict('train')
val_dict = get_person_image_dict('val')
test_dict = get_person_image_dict('test')

print(len(train_dict.keys()))
print(len(val_dict.keys()))
print(len(test_dict.keys()))

train_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.7, 1.02)),
    transforms.RandomAutocontrast(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomPerspective(),
    transforms.RandomGrayscale(),
    transforms.RandomInvert(p=0.1),
])

test_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



plt.figure(figsize=(5*4,2*4))

train_db = FaceDetectionDataset(train_dict, image_size=IMAGE_SIZE, transforms=train_data_transforms)
val_db = FaceDetectionDataset(val_dict, image_size=IMAGE_SIZE, transforms=test_data_transforms)

train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=os.cpu_count(), prefetch_factor=2)
val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), prefetch_factor=2)

i = 0
# for x1, x2, y in train_db:
#     plt.subplot(5,2,2*i+1)
#     plt.imshow(x1.permute(1,2,0))
#     plt.axis('off')
#     plt.title('same' if y == 1 else 'different')

#     plt.subplot(5,2,2*i+2)
#     plt.imshow(x2.permute(1,2,0))
#     plt.axis('off')
#     plt.title('same' if y == 1 else 'different')
    
#     i+= 1
#     if i == 5:
#         break
# plt.tight_layout()
# plt.show()
# print(train_dict)

model = FaceDetectionModel(hid_dim=512, out_dim=128).to(device)
loss_fn = ContrastiveLoss()
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model, optim = load_state('./model.pt', model, optim)

trainer = FaceDetectorTrainer(model, train_loader, val_loader, optim, loss_fn)
trainer.train_model(num_epochs=NUM_EPOCHS)