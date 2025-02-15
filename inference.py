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
from datasets import PersonIdentificationDataset
from models import PersonIdentificationModel
from losses import ContrastiveLoss
from trainer import PersonIdentifierTrainer


device = 'cuda' if torch.cuda.is_available() else 'cpu'


BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

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
    transforms.RandomRotation(15),
    # transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.7, 1.02)),
    transforms.RandomAutocontrast(),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.RandomPerspective(),
    transforms.RandomGrayscale(),
    transforms.RandomInvert(p=0.1),
])

test_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



plt.figure(figsize=(5*4,2*4))

train_db = PersonIdentificationDataset(train_dict, image_size=IMAGE_SIZE, transforms=train_data_transforms)
val_db = PersonIdentificationDataset(val_dict, image_size=IMAGE_SIZE, transforms=test_data_transforms)

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

model = PersonIdentificationModel(hid_dim=512, out_dim=128)
loss_fn = ContrastiveLoss()
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model, optim = load_state('model.pt', model, optim)


image1 = test_data_transforms(read_image('./im1.jpg', size=(IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0)
image2 = test_data_transforms(read_image('./im2.jpg', size=(IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0)
image3 = test_data_transforms(read_image('./im3.jpg', size=(IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0)
image4 = test_data_transforms(read_image('./im4.jpg', size=(IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0)
image5 = test_data_transforms(read_image('./im5.jpg', size=(IMAGE_SIZE, IMAGE_SIZE))).unsqueeze(0)
print(image4.shape)

import torch.nn.functional as F

model.eval()
with torch.inference_mode():
    d1 = model.inference(image1)
    d2 = model.inference(image2)
    d3 = model.inference(image3)
    d4 = model.inference(image4)
    d5 = model.inference(image5)
print(F.pairwise_distance(d1, d2))
print(F.pairwise_distance(d1, d3))
print(F.pairwise_distance(d1, d4))
print(F.pairwise_distance(d1, d5))