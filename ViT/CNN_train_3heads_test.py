import time
import datetime as datetime
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import timm
import clip
import torchvision.transforms as trans
import numpy as np
from datasets_3heads import *
import wandb

# 初始化W&B
wandb.init(project="Baselines_CNN")
config = wandb.config  # 设置配置

config.batch_size = 128
config.n_epochs = 10
config.learning_rate = 0.00005

torch.cuda.set_device(0)


class ViTModel(nn.Module):
    def __init__(self, vit_model, dim):
        super(ViTModel, self).__init__()
        self.classifiers = []
        self.conv = vit_model
        self.cls1 = nn.Sequential(nn.Linear(dim, 10))
        self.classifiers.append(self.cls1)
        self.cls2 = nn.Sequential(nn.Linear(dim, 6))
        self.classifiers.append(self.cls2)
        self.cls3 = nn.Sequential(nn.Linear(dim, 2))
        self.classifiers.append(self.cls3)
        # print(dim)
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
    def forward(self, x, ds, label):
        # print(ds)
        loss = 0
        feature = self.conv(x).type(torch.float32)
        for i in range(len(ds)):
            d = ds[i]
            y = label[i].unsqueeze(0)
            # print(y)
            # print(feature[i].shape)
            out = self.classifiers[d](feature[i].unsqueeze(0))
            # print(out)
            # print(out.shape)
            _, pred = torch.max(out.data, 1)
            loss = loss + self.criterion_cls(out, y)
            # print(out.shape)
            if i == 0 :
                prediction = pred
            else:
                prediction = torch.cat([prediction, pred], dim=0)      

        return prediction, loss


class ExGAN():
    def __init__(self):
        super(ExGAN, self).__init__()
        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs
        self.img_height = 224
        self.img_width = 224
        self.channels = 3
        self.model_name = "vit_tiny_patch16_224"
        self.task_name = "ViT"

        self.lr = config.learning_rate
        self.b1 = 0.5
        self.b2 = 0.999
        self.log_write = open("./results/log_CNN_results_in.txt", "w")

        self.img_shape = (self.channels, self.img_height, self.img_width)
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        # if self.model_name == 'ResNet18':
        #     self.model = models.resnet18(pretrained=True)
        #     num_ftrs = self.model.fc.in_features
        #     self.model.fc = nn.Sequential(nn.Linear(num_ftrs, self.c_dim),
        #                                   nn.LogSoftmax(dim=1))
        # self.model = models.alexnet(pretrained=True)
        #     num_ftrs = self.model.classifier[6].in_features
        #     self.model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, self.c_dim))
        
        ViT_model = models.resnet18(pretrained=True)
        ViT_model.head = nn.Sequential()
        # self.dim = ViT_model.fc.in_features
        self.dim = 1000
        self.model = ViTModel(ViT_model, self.dim)

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

        self.dataloader, self.test_dataloader = get_data(self.Transform, self.batch_size)

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

            for batch, (imgA, labelA, ds) in enumerate(self.dataloader, 1):
                X, y = imgA, labelA
                X, y = Variable(X.cuda()), Variable(y.cuda())
                ds = ds.numpy()
                numSample = numSample + y.size(0)
                pred, loss = self.model(X, ds, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(pred == y.data)
                
                wandb.log({"train_loss": loss, "epoch": epoch})

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
                        # 获取当前日期
                        current_date = datetime.datetime.now().strftime("%Y-%m%d")
                        # 保存模型权重
                        torch.save(self.model.state_dict(), "saved_models/model_{}_{}_{}_weights.pth".format(self.model_name, self.task_name, current_date))
                        # 打开日志文件以记录训练信息
                        log_write_record = open("./results/log_{}_{}_{}_train_record_in.txt".format(self.model_name, self.task_name, current_date), "w")
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
        running_loss = 0.0  # 累计测试损失
        numSample = 0
        s_idx = 0
        v_idx, v_pred, v_label = [], [], []
        
        dataset_corrects = np.zeros(3, dtype=int)  # 假设有三个数据集
        dataset_totals = np.zeros(3, dtype=int)

        for batch, (imgA, labelA, ds) in enumerate(dataloader, 1):
            X, y = imgA, labelA
            X, y = Variable(X.cuda()), Variable(y.cuda())
            ds = ds.numpy()
            numSample += y.size(0)

            pred, loss = self.model(X, ds, y)
            running_loss += loss.item() * X.size(0)  # 更新累计的测试损失
            running_corrects += torch.sum(pred == y.data)

            for k in range(len(y)):
                dataset_index = ds[k]
                dataset_totals[dataset_index] += 1
                if pred[k] == y[k]:
                    dataset_corrects[dataset_index] += 1

                v_idx.append(s_idx)
                v_pred.append(pred[k].item())
                v_label.append(y[k].item())
                s_idx += 1

            if batch % 100 == 0:
                current_loss = running_loss / numSample  # 计算当前平均测试损失
                current_acc = 100.0 * running_corrects / numSample
                print("Batch {}, Test Loss:{:.4f}, Test ACC:{:.4f}%".format(
                    batch, current_loss, current_acc))
                wandb.log({"batch_test_loss": current_loss, "batch_test_acc": current_acc, "batch": batch})

        epoch_loss = running_loss / numSample
        epoch_acc = 100.0 * running_corrects.item() / numSample
        print("Test Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss, epoch_acc))
        wandb.log({"test_loss": epoch_loss, "test_accuracy": epoch_acc})

        # 打印并记录每个数据集的准确度
        for i in range(3):
            if dataset_totals[i] > 0:
                dataset_acc = 100.0 * dataset_corrects[i] / dataset_totals[i]
                print("Dataset {} Acc: {:.4f}%".format(i + 1, dataset_acc))
                wandb.log({f"Dataset_{i+1}_accuracy": dataset_acc})

        time_end = time.time() - time_open
        print("程序运行时间:{}分钟...".format(int(time_end / 60)))

        return epoch_acc, v_idx, v_pred, v_label


def main():
    exgan = ExGAN()
    exgan.train()

if __name__ == "__main__":
    main()
