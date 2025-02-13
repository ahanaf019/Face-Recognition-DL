import torch
import torch.nn as nn
import random
from utils import read_image


class FaceDetectionDataset(nn.Module):
    def __init__(self, person_image_dict: dict, image_size=224, transforms = None):
        super().__init__()
        self.person_image_dict = person_image_dict
        self.transforms = transforms
        self.keys = list(person_image_dict.keys()) + list(person_image_dict.keys())
        self.image_size = image_size
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        person1 = self.person_image_dict[self.keys[index]]
        image1 = random.choice(person1)

        # Positive [Same person pair]
        if torch.randint(low=0, high=2, size=()).item() == 1:
            person2 = self.person_image_dict[self.keys[index]]
            label = 1
        
        # Negative [Different person pair]
        else:
            sampled = index
            while sampled == index:
                sampled = random.randint(0, len(self.keys) - 1)
            person2 = self.person_image_dict[self.keys[sampled]]
            label = 0
        image2 = random.choice(person2)


        image1 = read_image(image1, size=(self.image_size, self.image_size))
        image2 = read_image(image2, size=(self.image_size, self.image_size))

        if self.transforms is not None:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)

        return image1, image2, label