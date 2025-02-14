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

def get_image_bbox_dict(subset):
    image_base_dir = '~/Datasets/CelebA/img_celeba/test/10001/182659.jpg'
    partition_file = f'{os.environ["HOME"]}/Datasets/CelebA/Eval/list_eval_partition.txt'
    identity_file = f'{os.environ["HOME"]}/Datasets/CelebA/Anno/identity_CelebA.txt'
    bbox_file = f'{os.environ["HOME"]}/Datasets/CelebA/Anno/list_bbox_celeba.txt'
    # Read Partition File
    partition_df = pd.read_table(partition_file, sep=' ', header=None)
    partition_df.rename({
        0: 'filename',
        1: 'subset'
    }, axis=1, inplace=True)

    mapping = {
        0 : 'train',
        1 : 'val',
        2 : 'test'
    }
    partition_df['subset'] = partition_df['subset'].apply(lambda x: mapping[x])

    # Read Itentity File
    identity_df = pd.read_table(identity_file, sep=' ', header=None)
    identity_df.rename({
        0: 'filename',
        1: 'person_id'
    }, axis=1, inplace=True)

    # Merge Partition and Identity files
    df = pd.merge(partition_df, identity_df, on='filename')
    
    # Read and process BBox file
    with open(bbox_file, 'r') as f:
        lines = f.readlines()
    lines = [x.split() for x in lines]

    bbox_df = pd.DataFrame(lines[1:])
    bbox_df.rename({
        0: 'filename',
        1: 'x1',
        2: 'y1',
        3: 'width',
        4: 'height'
    }, axis=1, inplace=True)

    for column in bbox_df.columns[1:]:
        bbox_df[column] = bbox_df[column].astype(np.int32)
    # Merge into the final DataFrame
    df = pd.merge(df, bbox_df, on='filename')

    df = df[df['subset'] == subset]

    filenames = []
    for row in df.iterrows():
        filenames.append(f'{os.environ["HOME"]}/Datasets/CelebA/img_celeba/{subset}/{row[1]["person_id"]}/{row[1]["filename"]}')

    bbox = df.iloc[:, 3:].to_numpy()
    return filenames, bbox



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