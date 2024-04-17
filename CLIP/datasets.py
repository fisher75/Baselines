from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import os.path
import PIL.Image
from torchvision.datasets import MNIST
from PIL import Image
from random import shuffle
from torchvision.datasets import CIFAR100



class MNISTDataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/MNIST'
        self.sample_list = self.get_files(mode)
    def get_files(self, mode):
        sample_list = []
        if mode == 'train':
            alldata = MNIST(root=self.root, train=True)
        else:
            alldata = MNIST(root=self.root, train=False)

        data = alldata.data.detach().numpy()
        label = alldata.targets.detach().numpy()
        for i in range(data.shape[0]):
            sample = {}
            sam_data = data[i]
            sam_label = label[i]
            # sam_data = np.expand_dims(sam_data, 0)
            img = Image.fromarray(sam_data, mode='L').convert('RGB')
            sample['img'] = img
            sample['label'] = sam_label
            sample_list.append(sample)

        return sample_list

    def __getitem__(self, index):
        sample = self.sample_list[index % len(self.sample_list)]
        label = sample['label']
        img_sam = sample['img']
        img = self.transform(img_sam)

        return img, label

    def __len__(self):
        return len(self.sample_list)


class FlowerDataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/flower_data/arrange_data/'
        if mode == 'train':
            self.root = root + 'train-data/'
        elif mode == 'test':
            self.root = root + 'test-data/'
        self.files = os.listdir(self.root)

    def __getitem__(self, index):
        filename = self.files[index % len(self.files)]
        label = int(filename.split('_')[2])
        imgpath = self.root + filename
        img = self.transform(Image.open(imgpath))

        return img, label

    def __len__(self):
        return len(self.files)


class FERDataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/FER-dataseet/Oulu/img_augmentation/crop_img/'
        self.files = self.get_files(mode)
    def get_files(self, mode):
        fiels = []
        if mode == 'train':
            read_f = open(self.root + 'train_imgs1.txt', 'r')
        else:
            read_f = open(self.root + 'test_imgs1.txt', 'r')

        line = read_f.readline()
        while line:
            line = line.split()[0]
            fiels.append(line)
            line = read_f.readline()
        read_f.close()
        return fiels

    def __getitem__(self, index):
        filename = self.files[index % len(self.files)]
        imgname = filename.split('/')[-1]
        label = int(imgname.split('_')[1])
        imgpath = self.root + imgname
        img = self.transform(Image.open(imgpath))

        return img, label

    def __len__(self):
        return len(self.files)



class SAMDataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/Driver-abnormal-behaviour-recognition/SAM-DD/used-data/'
        sample_list = []
        read_f = open(self.root + mode + '_img.txt', 'r')
        print(self.root + mode + '_img.txt')
        line = read_f.readline()
        while line:
            line = line.split()[0]
            sample_list.append(self.root + line)
            line = read_f.readline()
        read_f.close()
        for i in range(5):
            shuffle(sample_list)
        self.sample_list = sample_list

    def __getitem__(self, index):
        filepath = self.sample_list[index % len(self.sample_list)]
        path_list = filepath.split('/')
        filename = path_list[-1]
        if filename.split('_')[0] == "sub":
            label = int(filename.split('_')[3])
        else:
            label = int(filename.split('_')[2])
        img = self.transform(Image.open(filepath))

        return img, label

    def __len__(self):
        return len(self.sample_list)


class TsingDogDataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/Tsinghua_dogs/used-data1/'
        sample_list = []
        self.cls_list = []
        read_f = open(self.root + mode + '_img_list.txt', 'r')
        print(self.root + mode + '_img_list.txt')
        line = read_f.readline()
        while line:
            line = line.split('\n')[0]
            sample_list.append(line)
            line = read_f.readline()
        read_f.close()
        read_l = open(self.root + 'label.txt', 'r')
        rlabel = read_l.readline()
        while rlabel:
            label = rlabel.split('\n')[0]
            self.cls_list.append(label)
            rlabel = read_l.readline()
        self.sample_list = sample_list

    def __getitem__(self, index):
        file = self.sample_list[index % len(self.sample_list)]
        cls = file.split('-')[0]
        label = self.cls_list.index(cls)
        filepath = self.root + 'images/' + file
        img = self.transform(Image.open(filepath))

        return img, label

    def __len__(self):
        return len(self.sample_list)

class FoodDataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/Food-101/food-101/'
        sample_list = []
        self.cls_list = []
        read_f = open(self.root + 'meta/' + mode + '.txt', 'r')
        print(self.root + 'meta/' + mode + '.txt')
        line = read_f.readline()
        while line:
            line = line.split('\n')[0] + '.jpg'
            sample_list.append(line)
            line = read_f.readline()
        read_f.close()
        read_l = open(self.root + 'meta/' + 'classes.txt', 'r')
        rlabel = read_l.readline()
        while rlabel:
            label = rlabel.split('\n')[0]
            self.cls_list.append(label)
            rlabel = read_l.readline()
        self.sample_list = sample_list

    def __getitem__(self, index):
        file = self.sample_list[index % len(self.sample_list)]
        cls = file.split('/')[0]
        label = self.cls_list.index(cls)
        filepath = self.root + 'images/' + file
        img = self.transform(Image.open(filepath))

        return img, label

    def __len__(self):
        return len(self.sample_list)

class LeafDataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.root = '/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/other-dataset/leaf-dataset/used-data/'
        sample_list = []
        self.cls_list = []
        read_f = open(self.root + mode + '_img_list.txt', 'r')
        print(self.root + mode + '_img_list.txt')
        line = read_f.readline()
        while line:
            line = line.split('\n')[0]
            sample_list.append(line)
            line = read_f.readline()
        read_f.close()
        read_l = open(self.root + 'label.txt', 'r')
        rlabel = read_l.readline()
        while rlabel:
            label = rlabel.split('\n')[0]
            self.cls_list.append(label)
            rlabel = read_l.readline()
        self.sample_list = sample_list

    def __getitem__(self, index):
        file = self.sample_list[index % len(self.sample_list)]
        cls = file.split('-img-')[0]
        label = self.cls_list.index(cls)
        filepath = self.root + 'images/' + file
        img = self.transform(Image.open(filepath))

        return img, label

    def __len__(self):
        return len(self.sample_list)

class Cifar100Dataset(Dataset):
    def __init__(self, transforms_=None, data=None):
        self.transform = transforms_
        self.sample_list = data

    def __getitem__(self, index):
        sample = self.sample_list[index % len(self.sample_list)]
        label = sample['label']
        img_sam = sample['img']
        img = PIL.Image.fromarray(img_sam)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.sample_list)

def get_dataloader(preprocess, batch_size, task, mode, shuffle):
    if task == 'food':
        data = DataLoader(
            FoodDataset(transforms_=preprocess, mode=mode),
            batch_size=batch_size, shuffle=shuffle, num_workers=8)
    elif task == 'dog':
        data = DataLoader(
        TsingDogDataset(transforms_=preprocess, mode=mode),
        batch_size=batch_size, shuffle=shuffle, num_workers=8)
    elif task == 'distraction':
        data = DataLoader(
        SAMDataset(transforms_=preprocess, mode=mode),
        batch_size=batch_size, shuffle=shuffle, num_workers=8)
    elif task == 'expression':
        data = DataLoader(
        FERDataset(transforms_=preprocess, mode=mode),
        batch_size=batch_size, shuffle=shuffle, num_workers=8)
    elif task == 'flower':
        data = DataLoader(
        FlowerDataset(transforms_=preprocess, mode=mode),
        batch_size=batch_size, shuffle=shuffle, num_workers=8)
    elif task == 'mnist':
        data = DataLoader(
        MNISTDataset(transforms_=preprocess, mode=mode),
        batch_size=batch_size, shuffle=shuffle, num_workers=8)
    elif task == 'leaf':
        data = DataLoader(
        LeafDataset(transforms_=preprocess, mode=mode),
        batch_size=batch_size, shuffle=shuffle, num_workers=8)
    elif task == 'cifar-100':
        root = os.path.expanduser("~/.cache")
        if mode == 'train':
            data = DataLoader(
                CIFAR100(root, download=True, train=True, transform=preprocess),
                batch_size=batch_size, shuffle=shuffle, num_workers=8)
        else:
            data = DataLoader(
                CIFAR100(root, download=True, train=False, transform=preprocess),
                batch_size=batch_size, shuffle=shuffle, num_workers=8)

    return data


def get_data(preprocess, batch_size, task_list, task):
    train_data = get_dataloader(preprocess, batch_size, task_list[task], 'train', True)
    test_data = get_dataloader(preprocess, batch_size, task_list[task], 'test', False)

    return train_data, test_data