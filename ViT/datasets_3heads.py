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
        self.root = os.path.join(data_root, mode)
        self.files = [f for f in os.listdir(self.root) if f.endswith('.jpg')]  # 确保只处理.jpg文件

    def __getitem__(self, index):
        filename = self.files[index % len(self.files)]
        try:
            ds_num = int((filename.split('ds')[-1]).split('_')[0]) - 1
            label = int((filename.split('cls')[-1]).split('_')[0])
            imgpath = os.path.join(self.root, filename)
            img = self.transform(Image.open(imgpath))
            return img, label, ds_num
        except Exception as e:
            # 处理文件名解析错误或文件读取错误
            print(f"Error processing file {filename}: {e}")
            return None, None, None

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