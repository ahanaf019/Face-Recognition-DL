import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset
from utils import read_image, resize_image_bbox
import cv2

class PersonIdentificationDataset(Dataset):
    def __init__(self, person_image__bbox_dict: dict, image_size=224, transforms = None):
        super().__init__()
        self.person_image_dict = person_image__bbox_dict
        self.transforms = transforms
        self.keys = list(person_image__bbox_dict.keys()) + list(person_image__bbox_dict.keys())
        self.image_size = image_size
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        person1 = self.person_image_dict[self.keys[index]]
        # print(person1)
        image_bbox_pair1 = random.choice(person1)

        # Positive [Same person pair]
        if random.random() > 0.5:
            person2 = self.person_image_dict[self.keys[index]]
            label = 1
        
        # Negative [Different person pair]
        else:
            negative_person = random.choice([k for k in self.keys if k != self.keys[index]])
            person2 = self.person_image_dict[negative_person]
            label = 0
        image_bbox_pair2 = random.choice(person2)


        image1 = read_image(image_bbox_pair1[0], size=None)
        image2 = read_image(image_bbox_pair2[0], size=None)

        x1, y1, w1, h1 = image_bbox_pair1[1]
        x2, y2, w2, h2 = image_bbox_pair2[1]
        image1 = image1[y1:y1 + h1, x1:x1 + w1]
        image2 = image2[y2:y2 + h2, x2:x2 + w2]
        
        image1 = cv2.resize(image1, (self.image_size, self.image_size))
        image2 = cv2.resize(image2, (self.image_size, self.image_size))

        if self.transforms is not None:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)

        return image1, image2, torch.tensor(label, dtype=torch.float32)
    

class FaceBBoxDataset(Dataset):
    def __init__(self, paths, bboxes, image_size=224, transforms=None):
        super().__init__()
        self.paths = paths
        self.bboxes = bboxes
        self.image_size = image_size
        self.transforms = transforms
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        bbox = self.bboxes[index]

        image = read_image(path, size=None)
        image, bbox = resize_image_bbox(image, bbox, new_size=(self.image_size, self.image_size))
        bbox = [x / self.image_size for x in bbox]
        
        if self.transforms is not None:
            image = self.transforms(image)
        return image, torch.tensor(bbox)