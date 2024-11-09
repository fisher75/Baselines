'''将Exchange-GAN和Star-GAN相结合, 使该结构能够适用于非匹配数据集
对中心向量之间的最小距离设置了阈值, 并将中心向量的更新的学习率调整为0.001
将损失函数中分类损失的权重调整为1,并去掉了生成器对抗损失前面的权重3
'''

import argparse
import os
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import sys
import scipy.io

import torchvision
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision import models

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchinfo import summary

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import timm
from dataset import *

torch.cuda.set_device(1)


class ExGAN():
    def __init__(self, model_name):
        super(ExGAN, self).__init__()
        self.batch_size = 24
        self.n_epochs = 6
        self.img_height = 224
        self.img_width = 224
        self.channels = 3
        self.c_dim = 3
        self.model_name = model_name

        self.lr = 0.00005
        self.b1 = 0.5
        self.b2 = 0.999
        self.log_write = open("./results/log_%s_results.txt" % self.model_name, "w")

        self.img_shape = (self.channels, self.img_height, self.img_width)
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        if self.model_name == 'VGG16':
            self.model = models.vgg16_bn(pretrained=True)
            # for param in self.model.parameters():
            #     param.requires_grad = False
            '''定义新的全连接层并重新赋值给 model.classifier，重新设计分类器的结构，此时 parma.requires_grad 会被默认重置为 True'''
            self.model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                                   torch.nn.ReLU(),
                                                   torch.nn.Dropout(p=0.5),
                                                   torch.nn.Linear(4096, 4096),
                                                   torch.nn.ReLU(),
                                                   torch.nn.Dropout(p=0.5),
                                                   torch.nn.Linear(4096, self.c_dim))

        elif self.model_name == 'VGG11':
            self.model = models.vgg11_bn(pretrained=True)
            # for param in self.model.parameters():
            #     param.requires_grad = False
            '''定义新的全连接层并重新赋值给 model.classifier，重新设计分类器的结构，此时 parma.requires_grad 会被默认重置为 True'''
            self.model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                                   torch.nn.ReLU(),
                                                   torch.nn.Dropout(p=0.5),
                                                   torch.nn.Linear(4096, 4096),
                                                   torch.nn.ReLU(),
                                                   torch.nn.Dropout(p=0.5),
                                                   torch.nn.Linear(4096, self.c_dim))

        elif self.model_name == 'ResNet18':
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, self.c_dim),
                                          nn.LogSoftmax(dim=1))
        elif self.model_name == 'AlexNet':
            self.model = models.alexnet(pretrained=True)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, self.c_dim))

        elif self.model_name == "EfficientNet":          ##  ["EfficientNet", "Mobilenet2", "Mobilenet3", "ShuffleNet_v2_x1_0", "ShuffleNet_v2_x1_0", "SqueezeNet1_0"]
            self.model = models.efficientnet_v2_m(pretrained=True)
            self.model.classifier[1] = torch.nn.Linear(1280, self.c_dim)

        elif self.model_name == "Mobilenet2":
            self.model = models.mobilenet_v2(pretrained=True)
            self.model.classifier[1] = torch.nn.Sequential(
                torch.nn.Linear(1280, self.c_dim)
            )

        elif self.model_name == "ViT":
            self.img_height = 224
            self.img_width = 224
            self.model = timm.create_model("vit_tiny_patch16_224", pretrained=True,
                                          num_classes=5)
            self.model.head = torch.nn.Linear(192, self.c_dim)

        elif self.model_name == "GoogleNet":
            self.img_height = 224
            self.img_width = 224
            self.model = models.inception_v3(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.c_dim)

        elif model_name == "ShuffleNet_v2_x2_0":
            self.model = models.shufflenet_v2_x2_0(pretrained=True)
            self.model.fc = torch.nn.Linear(2048, self.c_dim)

        elif model_name == "ShuffleNet_v2_x1_0":
            self.model = models.shufflenet_v2_x1_0(pretrained=True)
            self.model.fc = torch.nn.Linear(1024, self.c_dim)

        elif model_name == "SqueezeNet1_0":
            self.model = models.squeezenet1_0(pretrained=True)
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.5, inplace=False),
                torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1)),
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        elif self.model_name == "DenseNet":
            self.model = models.densenet121(pretrained=True)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(num_ftrs, self.c_dim)

        elif self.model_name == "Swin-Transformer":
            self.img_height = 224
            self.img_width = 224
            self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True,
                                          num_classes=5)
            self.model.head.fc = torch.nn.Linear(768, self.c_dim)

        self.model = self.model.cuda()
        # total = sum([param.nelement() for param in self.model.parameters()])
        # print("%s-Number of parameter: %.4fM" % (self.model_name, total / 1e6))
        summary(self.model, (64, 3, 224, 224))

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))


        self.Transform = [
            transforms.Resize((self.img_height, self.img_width), Image.BICUBIC),
            transforms.ToTensor(),
            ]
        # self.Transform  = [
        #     transforms.Resize(299),
        #     transforms.CenterCrop(299),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ]

        self.dataloader = DataLoader(
            LaryDataset(root='/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/Lary-detection-dataset/',
                           transforms_=self.Transform, mode='train'),
            batch_size=64,
            shuffle=True,
            num_workers=8,
            )

        self.test_dataloader = DataLoader(
            LaryDataset(root='/media/baiyang/02248a30-d286-4856-8662-fd2c6a68eba6/automan/dataset/Lary-detection-dataset/',
                           transforms_=self.Transform, mode='test'),
            batch_size=64,
            shuffle=True,
            num_workers=8,
            )



    def train(self):
        print('Train on the %s model' % self.model_name)
        time_open = time.time()
        phase = 'train'
        best_accuracy = 0
        average_accuracy = 0
        num_test = 0
        for epoch in range(self.n_epochs):
            print("-" * 10)
            print("Training...")
            self.model.train(True)

            running_loss = 0.0
            running_corrects = 0
            # cxq = 1
            # for batch, data in enumerate(dataloader[phase], 1):
            numSample = 0

            for batch, (imgA, labelA) in enumerate(self.dataloader, 1):
                X, y = imgA, labelA
                X, y = Variable(X.cuda()), Variable(y.cuda())
                numSample = numSample + y.size(0)

                y_pred = self.model(X)

                _, pred = torch.max(y_pred.data, 1)

                self.optimizer.zero_grad()

                loss = self.criterion_cls(y_pred, y)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(pred == y.data)

                if batch % 10 == 0:
                    print("Epoch {}/{}, Batch {}, Train Loss:{:.4f},Train ACC:{:.4f}%".format(
                        epoch, self.n_epochs - 1, batch, running_loss / batch, 100.0 * running_corrects / numSample
                    ))

                if batch % 200 == 0:
                    testAcc, vidx, vpred, vlable = self.test()
                    num_test = num_test + 1
                    self.log_write.write(str(epoch) + "    " + str(batch) + "    " + str(testAcc) + "\n")
                    average_accuracy = average_accuracy + testAcc
                    if testAcc > best_accuracy:
                        best_accuracy = testAcc
                        torch.save(self.model.state_dict(), "saved_models/model_%s_weights.pth" % self.model_name)
                        log_write_record = open("log_%s_train_record.txt" % self.model_name, "w")
                        for r in range(len(vidx)):
                            log_write_record.write(str(vidx[r]) + "  " + str(vpred[r]) + "  " + str(vlable[r]) + "\n")
                        log_write_record.close()

            epoch_loss = running_loss * 16 / numSample
            epoch_acc = 100.0 * running_corrects / numSample

            print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase, epoch_loss, epoch_acc))

            time_end = time.time() - time_open
            print("程序运行时间:{}分钟...".format(int(time_end / 60)))

            testAcc, vidx, vpred, vlable = self.test()
            num_test = num_test + 1
            self.log_write.write(str(epoch) + "    " + str(batch) + "    " + str(testAcc) + "\n")
            average_accuracy = average_accuracy + testAcc
            if testAcc > best_accuracy:
                best_accuracy = testAcc
                torch.save(self.model.state_dict(), "saved_models/model_%s_weights.pth" % self.model_name)
                log_write_record = open("log_%s_train_record.txt" % self.model_name, "w")
                for r in range(len(vidx)):
                    log_write_record.write(str(vidx[r]) + "  " + str(vpred[r]) + "  " + str(vlable[r]) + "\n")
                log_write_record.close()

        average_accuracy = average_accuracy / num_test
        print('The best accuracy is: ' + str(best_accuracy) + "\n")
        print('The average accuracy is: ' + str(average_accuracy) + "\n")
        self.log_write.write('The best accuracy is: ' + str(best_accuracy) + "\n")
        self.log_write.write('The average accuracy is: ' + str(average_accuracy) + "\n")
        self.log_write.close()


    def test(self):
        time_open = time.time()
        running_corrects = 0
        numSample = 0
        s_idx = 0
        v_idx, v_pred, v_label = [], [], []

        if os.path.exists("saved_models/model_%s_weights.pth" % self.model_name):
            print("Load exit modle.")
            self.model.load_state_dict(torch.load("saved_models/model_%s_weights.pth" % self.model_name))
        else:
            self.train()
            self.model.load_state_dict(torch.load("saved_models/model_%s_weights.pth" % self.model_name))

        for batch, (imgA, labelA) in enumerate(self.test_dataloader, 1):
            X, y = imgA, labelA
            X, y = Variable(X.cuda()), Variable(y.cuda())
            numSample = numSample + y.size(0)

            y_pred = self.model(X)
            _, pred = torch.max(y_pred.data, 1)
            running_corrects += torch.sum(pred == y.data)

            for k in range(len(labelA)):
                v_idx.append(s_idx)
                v_pred.append(pred[k].item())
                v_label.append(labelA[k].item())
                s_idx = s_idx + 1
            # print("Batch {}, Test ACC:{:.4f}%".format(
            #     batch, 100.0 * running_corrects / numSample))

        epoch_acc = 100.0 * running_corrects.item() / numSample
        print("{} Acc:{:.4f}%".format('test', epoch_acc))

        time_end = time.time() - time_open
        print("程序运行时间:{}分钟...".format(int(time_end / 60)))

        FPS = 1/(time_end / numSample)
        print("The average fps is: %f" % FPS)

        return epoch_acc, v_idx, v_pred, v_label


def main():
    # models_list = ['AlexNet']
    # models_list = ["ShuffleNet_v2_x1_0", "ShuffleNet_v2_x1_0", "SqueezeNet1_0"]
    models_list = ['AlexNet', 'VGG11', 'ResNet18', 'DenseNet', 'Mobilenet2', 'ShuffleNet_v2_x1_0', 'ViT', 'Swin-Transformer']
    for model in models_list:
        exgan = ExGAN(model)
        exgan.test()

if __name__ == "__main__":
    main()
