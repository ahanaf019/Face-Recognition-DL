import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from utils import *
from datasets import PersonIdentificationDataset
from models import PersonIdentificationModel
from losses import ContrastiveLoss
from trainer import PersonIdentifierTrainer


device = 'cuda' if torch.cuda.is_available() else 'cpu'


BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE = 5e-3
NUM_EPOCHS = 70


train_dict = get_person_image_bbox_dict('train')
val_dict = get_person_image_bbox_dict('val')
test_dict = get_person_image_bbox_dict('test')


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

model = PersonIdentificationModel(hid_dim=512, out_dim=256).to(device)
loss_fn = ContrastiveLoss(margin=3.0)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-6)


trainer = PersonIdentifierTrainer(model, train_loader, val_loader, optim, loss_fn, save_filename=f'./checkpoints/{model.__class__.__name__}.pt')
trainer.train_model(num_epochs=NUM_EPOCHS, lr_reduce_patience=5, early_stop_patience=16)