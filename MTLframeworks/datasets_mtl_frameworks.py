from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import PIL.Image
from PIL import Image
from torchvision import transforms

class FERDataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        data_root = '/home/users/ntu/chih0001/scratch/data/mixed'  # 设置数据路径
        self.root = os.path.join(data_root, mode)  # 根据模式选择目录
        self.files = os.listdir(self.root)  # 读取文件列表

    def __getitem__(self, index):
        filename = self.files[index % len(self.files)]
        # 获取数据集编号和标签
        ds_num = int((filename.split('ds')[-1]).split('_')[0]) - 1
        label = int((filename.split('cls')[-1]).split('_')[0])
        
        # 加载图像
        imgpath = os.path.join(self.root, filename)
        img = Image.open(imgpath).convert("RGB")  # 确保图像是RGB模式
        img = self.transform(img)  # 应用预处理
        
        # 返回图像、标签和数据集编号，适用于多任务框架
        return img, label, ds_num

    def __len__(self):
        return len(self.files)


def get_dataloader(preprocess, batch_size, mode, shuffle):
    return DataLoader(
        FERDataset(transforms_=preprocess, mode=mode),
        batch_size=batch_size, shuffle=shuffle, num_workers=8
    )


def get_data(preprocess, batch_size):
    train_data = get_dataloader(preprocess, batch_size, 'train', True)  # 随机打乱训练数据
    test_data = get_dataloader(preprocess, batch_size, 'test', False)  # 测试数据按顺序加载
    return train_data, test_data
