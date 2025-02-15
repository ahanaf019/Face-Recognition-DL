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
from utils import *
from datasets import PersonIdentificationDataset
from models import PersonIdentificationModel
from losses import ContrastiveLoss
from trainer import PersonIdentifierTrainer


device = 'cuda' if torch.cuda.is_available() else 'cpu'


BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE = 1e-4
NUM_EPOCHS = 70

def get_person_image_bbox_dict(subset):
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
        filenames.append(f'{os.environ["HOME"]}/Datasets/CelebA/img_celeba/{subset}/{str(row[1]["person_id"])}/{row[1]["filename"]}')

    def process_row(row):
        row['filename'] = f'{os.environ["HOME"]}/Datasets/CelebA/img_celeba/{row["subset"]}/{row["person_id"]}/{row["filename"]}'
        return row

    person_dict = {}
    group = df.groupby('person_id')
    for person, grouped_df in group:
        grouped_df = grouped_df.apply(process_row, axis=1)
        filenames = grouped_df['filename'].tolist()
        bboxes = grouped_df.iloc[:, 3:].to_numpy()
        person_dict[person] = list(zip(filenames, bboxes))
    return person_dict



train_dict = get_person_image_bbox_dict('train')
val_dict = get_person_image_bbox_dict('val')
test_dict = get_person_image_bbox_dict('test')


def validate(dict):
    for key in dict.keys():
        pairs = dict[key]
        for path, bbox in pairs:
            if not os.path.exists(path):
                print('ERROR:',path)

print('Train')
validate(train_dict)
print('Val')
validate(val_dict)
print('Test')
validate(test_dict)



print(len(train_dict.keys()))
print(len(val_dict.keys()))
print(len(test_dict.keys()))

train_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomRotation(20),
    transforms.RandomAutocontrast(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomPerspective(),
    transforms.RandomGrayscale(),
    transforms.RandomInvert(p=0.1),
])

test_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




train_db = PersonIdentificationDataset(train_dict, image_size=IMAGE_SIZE, transforms=train_data_transforms)
val_db = PersonIdentificationDataset(val_dict, image_size=IMAGE_SIZE, transforms=test_data_transforms)

train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=os.cpu_count(), prefetch_factor=2)
val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), prefetch_factor=2)

# plt.figure(figsize=(5*4,2*4))
# i = 0
# for x1, x2, y in val_db:
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

model = PersonIdentificationModel(hid_dim=512, out_dim=128).to(device)
loss_fn = ContrastiveLoss(margin=2.0)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


trainer = PersonIdentifierTrainer(model, train_loader, val_loader, optim, loss_fn, save_filename=f'./checkpoints/{model.__class__.__name__}.pt')

trainer.load_state(f'./checkpoints/PersonIdentificationModel.pt')

trainer.train_model(num_epochs=NUM_EPOCHS)