from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import os.path
import PIL.Image
from torchvision.datasets import MNIST
from PIL import Image
from random import shuffle
from torchvision.datasets import CIFAR100


class FERDataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        data_root = '/home/users/ntu/chih0001/scratch/data/mixed'
        self.root = data_root + "/" + mode + "/"
        self.files = os.listdir(self.root)

    def __getitem__(self, index):
        add_label = [0, 10, 16]
        filename = self.files[index % len(self.files)]
        ds_num = int((filename.split('ds')[-1]).split('_')[0]) - 1
        label = int((filename.split('cls')[-1]).split('_')[0]) + add_label[ds_num]
        imgpath = self.root + filename
        img = self.transform(Image.open(imgpath))

        return img, label, ds_num

    def __len__(self):
        return len(self.files)


def get_dataloader(preprocess, batch_size, mode, shuffle):
    data = DataLoader(
        FERDataset(transforms_=preprocess, mode=mode),
        batch_size=batch_size, shuffle=shuffle, num_workers=8)

    return data


def get_data(preprocess, batch_size):
    train_data = get_dataloader(preprocess, batch_size, 'train', True)
    test_data = get_dataloader(preprocess, batch_size, 'test', False)

    return train_data, test_data