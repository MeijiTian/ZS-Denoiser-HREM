import torch
import utils
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image


class ImgTrain(data.Dataset):
    def __init__(self, x1_data,patch_size,istransform):
        self.x1 = x1_data
        self.patch_size = patch_size
        self.istransform = istransform
        assert patch_size <= x1_data.shape[1]
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, item):
        x1 = self.x1[item]

        x1 = Image.fromarray(x1)
        if self.istransform == True:
            x1 = self.transform(x1)

        return x1

    def get_transform(self):
        transform = []
        transform.append(transforms.RandomHorizontalFlip()) if self.istransform else None
        transform.append(transforms.RandomVerticalFlip()) if self.istransform else None
        transform.append(transforms.RandomRotation(20)) if self.istransform else None
        transform.append(transforms.RandomCrop(self.patch_size)) if self.istransform else None
        transform.append(transforms.ToTensor())
        return transforms.Compose(transform)


class ImgVal(data.Dataset):
    def __init__(self, x1_data):
        self.x1 = x1_data

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, item):
        x1 = self.x1[item]
        return x1


def loader_train(x1,patch_size,batch_size):
    return data.DataLoader(
        dataset=ImgTrain(x1,patch_size, istransform=True),
        batch_size = batch_size,
        shuffle = True
    )

def loader_val(x1,batch_size):
    return data.DataLoader(
        dataset=ImgVal(x1),
        batch_size = batch_size,
        shuffle = False
    )

