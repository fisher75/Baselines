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
        data_root = '/home/users/ntu/chih0001/scratch/data/mixed' # 这个是全数据train
        # data_root = '/home/users/ntu/chih0001/scratch/data/mixed' # 这个是fewshot的10shot用的
        self.root = data_root + "/" + mode + "/" # 根据模式选择目录
        self.files = os.listdir(self.root) # 读取文件列表

    def __getitem__(self, index):
        filename = self.files[index % len(self.files)] # 获取文件名
        ds_num = int((filename.split('ds')[-1]).split('_')[0]) - 1 # 解析数据集编号
        label = int((filename.split('cls')[-1]).split('_')[0]) # 解析标签
        imgpath = self.root + filename
        img = self.transform(Image.open(imgpath)) # 读取并变换图像

        return img, label, ds_num

    def __len__(self):
        return len(self.files) # 返回数据集长度


def get_dataloader(preprocess, batch_size, mode, shuffle):
    data = DataLoader(
        FERDataset(transforms_=preprocess, mode=mode),
        batch_size=batch_size, shuffle=shuffle, num_workers=8)

    return data


def get_data(preprocess, batch_size):
    train_data = get_dataloader(preprocess, batch_size, 'train', True) # mode='train'数据是随机打乱的。
    test_data = get_dataloader(preprocess, batch_size, 'test', False) # mode='test'数据是按顺序加载的。

    return train_data, test_data