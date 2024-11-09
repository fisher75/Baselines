import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms



class LaryDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        if mode == 'train':
            self.root = root + 'train-data/'
        elif mode == 'test':
            self.root = root + 'test-data/'
        self.files = os.listdir(self.root)

    def __getitem__(self, index):
        filename = self.files[index % len(self.files)]
        label = int(filename.split('-')[1])-1
        imgpath = self.root + filename

        img = self.transform(Image.open(imgpath))
        img = img * 2 - 1

        return img, label

    def __len__(self):
        return len(self.files)

