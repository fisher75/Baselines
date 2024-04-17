'''将Exchange-GAN和Star-GAN相结合, 使该结构能够适用于非匹配数据集
对中心向量之间的最小距离设置了阈值, 并将中心向量的更新的学习率调整为0.001
将损失函数中分类损失的权重调整为1,并去掉了生成器对抗损失前面的权重3
'''


import time
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import timm
import clip
import torchvision.transforms as trans
from datasets import *



torch.cuda.set_device(1)


class ViTModel(nn.Module):
    def __init__(self, vit_model, c_dim, dim):
        super(ViTModel, self).__init__()
        self.conv = vit_model
        self.classifier = nn.Sequential(nn.Linear(dim, c_dim),
                                        nn.LogSoftmax(dim=1))

    def forward(self, x):
        feature = self.conv(x).type(torch.float32)
        # print(feature.shape)
        # feature = feature.view(feature.size(0), -1)
        # print(feature.shape)
        out = self.classifier(feature)

        return out


class ExGAN():
    def __init__(self):
        super(ExGAN, self).__init__()
        self.batch_size = 24
        self.n_epochs = 10
        self.img_height = 224
        self.img_width = 224
        self.channels = 3
        self.model_name = "ViT"
        self.task = 5
        self.num_dim_list = [10, 100, 38, 101, 100, 10, 6]
        self.task_list = ['mnist', 'cifar-100', 'leaf', 'food', 'dog', 'distraction', 'expression']
        self.c_dim = self.num_dim_list[self.task]
        self.task_name = self.task_list[self.task]

        self.lr = 0.00005
        self.b1 = 0.5
        self.b2 = 0.999
        self.log_write = open("./results/log_%s_%s_results_in.txt" % (self.model_name, self.task_name), "w")

        self.img_shape = (self.channels, self.img_height, self.img_width)
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        ViT_model = timm.create_model("vit_tiny_patch16_224", pretrained=True,
                                       num_classes=5)
        ViT_model.head = nn.Sequential()
        self.dim = 192
        self.model = ViTModel(ViT_model, self.c_dim, self.dim)

        self.model = self.model.cuda()
        total = sum([param.nelement() for param in self.model.parameters()])
        print("%s-Number of parameter: %.4fM" % (self.model_name, total / 1e6))

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.Transform = trans.Compose([
            transforms.Resize(self.img_height),
            transforms.CenterCrop(self.img_height),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.dataloader, self.test_dataloader = get_data(self.Transform, self.batch_size, self.task_list, self.task)

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
                    testAcc, vidx, vpred, vlable = self.test(self.test_dataloader)
                    num_test = num_test + 1
                    self.log_write.write(str(epoch) + "    " + str(batch) + "    " + str(testAcc) + "\n")
                    average_accuracy = average_accuracy + testAcc
                    if testAcc > best_accuracy:
                        best_accuracy = testAcc
                        torch.save(self.model.state_dict(), "saved_models/model_%s_%s_weights.pth" % (self.model_name, self.task_name))
                        log_write_record = open("./results/log_%s_%s_train_record_in.txt" % (self.model_name, self.task_name), "w")
                        for r in range(len(vidx)):
                            log_write_record.write(str(vidx[r]) + "  " + str(vpred[r]) + "  " + str(vlable[r]) + "\n")
                        log_write_record.close()

            epoch_loss = running_loss * 16 / numSample
            epoch_acc = 100.0 * running_corrects / numSample

            print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase, epoch_loss, epoch_acc))

            time_end = time.time() - time_open
            print("程序运行时间:{}分钟...".format(int(time_end / 60)))

            testAcc, vidx, vpred, vlable = self.test(self.test_dataloader)
            num_test = num_test + 1
            self.log_write.write(str(epoch) + "    " + str(batch) + "    " + str(testAcc) + "\n")
            average_accuracy = average_accuracy + testAcc
            if testAcc > best_accuracy:
                best_accuracy = testAcc
                torch.save(self.model.state_dict(), "saved_models/model_%s_%s_weights.pth" % (self.model_name, self.task_name))
                log_write_record = open("./results/log_%s_%s_train_record_in.txt" % (self.model_name, self.task_name), "w")
                for r in range(len(vidx)):
                    log_write_record.write(str(vidx[r]) + "  " + str(vpred[r]) + "  " + str(vlable[r]) + "\n")
                log_write_record.close()

        average_accuracy = average_accuracy / num_test
        print('The best accuracy is: ' + str(best_accuracy) + "\n")
        print('The average accuracy is: ' + str(average_accuracy) + "\n")
        self.log_write.write('The best accuracy is: ' + str(best_accuracy) + "\n")
        self.log_write.write('The average accuracy is: ' + str(average_accuracy) + "\n")
        self.log_write.close()


    def test(self, dataloader):
        time_open = time.time()
        running_corrects = 0
        numSample = 0
        s_idx = 0
        v_idx, v_pred, v_label = [], [], []

        for batch, (imgA, labelA) in enumerate(dataloader, 1):
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

            if batch % 100 == 0:
                print("Batch {}, Test ACC:{:.4f}%".format(
                    batch, 100.0 * running_corrects / numSample))

        epoch_acc = 100.0 * running_corrects.item() / numSample

        print("{} Acc:{:.4f}%".format('test', epoch_acc))

        time_end = time.time() - time_open
        print("程序运行时间:{}分钟...".format(int(time_end / 60)))

        return epoch_acc, v_idx, v_pred, v_label


def main():
    exgan = ExGAN()
    exgan.train()

if __name__ == "__main__":
    main()
