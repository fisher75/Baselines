import time
import datetime as datetime
from sklearn.linear_model import LogisticRegression
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import timm
import clip
import torchvision.transforms as trans
import numpy as np
from datasets import *
import wandb
from tqdm import tqdm

# 初始化W&B
wandb.init(project="Baselines_ViT")
config = wandb.config  # 设置配置

config.batch_size = 24
config.n_epochs = 10
config.learning_rate = 0.00005

torch.cuda.set_device(0)


class ViTModel(nn.Module):
    def __init__(self, vit_model, dim):
        super(ViTModel, self).__init__()
        self.conv = vit_model  # 使用传入的Vision Transformer模型作为特征提取器
        self.classifier = nn.Sequential(nn.Linear(dim, 18))  # 定义一个线性分类器

    def forward(self, x, ds=None):
        # 将输入图像x通过Vision Transformer模型进行特征提取
        feature = self.conv(x).type(torch.float32)

        # 如果提供了数据集索引ds，并且需要根据ds进行特定处理，可以在这里添加逻辑
        # 例如，使用不同的分类器处理不同的数据集
        # if ds is not None:
        #     # 假设self.classifiers是一个包含不同分类器的列表，根据ds选择相应的分类器
        #     out = self.classifiers[ds](feature)
        # else:
        #     # 如果没有提供ds或不需要使用它，就使用默认的分类器
        #     out = self.classifier(feature)

        # 在这个示例中，我们不使用ds，直接使用单一的分类器
        out = self.classifier(feature)

        return out



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
        self.log_write = open("./results/log_%s_results_in.txt" % (self.model_name), "w")

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

    

    def get_features(self, dataset):
        all_features = []
        all_labels = []
        all_datasets = []

        with torch.no_grad():
            for batch, (imgA, label, ds) in enumerate(dataset, 1):
                X, y, ds_tensor = imgA.cuda(), label, ds.cuda()  # 确保ds也在正确的设备上
                features = self.model(X, ds_tensor)  # 现在传入ds

                all_features.append(features)
                all_labels.append(y)
                all_datasets.extend(ds.numpy())  # 收集原始ds信息用于后续统计

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy(), all_datasets

    
    def zero_shot_test(self):
        # 假设get_features方法已返回训练特征、标签和数据集编号
        train_features, train_labels, train_datasets = self.get_features(self.dataloader)
        test_features, test_labels, test_datasets = self.get_features(self.test_dataloader)

        # 使用逻辑回归进行训练
        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
        classifier.fit(train_features, train_labels)

        # 使用训练好的分类器进行预测
        predictions = classifier.predict(test_features)
        total_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
        print(f"Total Accuracy = {total_accuracy:.3f}%")
        wandb.log({"total_accuracy": total_accuracy})  # 在WandB中记录总准确度

        # 初始化数据集准确度统计
        dataset_accuracies = [0, 0, 0]
        dataset_counts = [0, 0, 0]

        # 计算每个数据集的准确度
        for pred, label, dataset in zip(predictions, test_labels, test_datasets):
            dataset_index = dataset  # 确保这里的dataset变量正确表示数据集索引
            dataset_counts[dataset_index] += 1
            if pred == label:
                dataset_accuracies[dataset_index] += 1

        # 打印并记录每个数据集的准确度
        for i in range(3):
            if dataset_counts[i] > 0:
                acc = (dataset_accuracies[i] / dataset_counts[i]) * 100
                print(f"Accuracy for Dataset {i+1}: {acc:.3f}%")
                wandb.log({f"accuracy_dataset_{i+1}": acc})

        # 使用tqdm显示测试进度
        for batch in tqdm(self.test_dataloader, desc="Zero-shot Testing"):
            img, label, dataset = batch  # 假设这些变量已经正确从dataloader解包
            # 这里不实际运行模型，因为模型推理已在上方的逻辑中完成
            
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
                y_pred = self.model(X, ds)
                _, pred = torch.max(y_pred.data, 1)

                self.optimizer.zero_grad()
                loss = self.criterion_cls(y_pred, y)
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
        numSample = 0
        s_idx = 0
        v_idx, v_pred, v_label = [], [], []
        
        # 添加计数器以跟踪每个数据集的正确预测数和样本总数
        dataset_corrects = np.zeros(3, dtype=int)  # 假设有三个数据集
        dataset_totals = np.zeros(3, dtype=int)

        for batch, (imgA, labelA, ds) in enumerate(dataloader, 1):
            X, y = imgA, labelA
            X, y = Variable(X.cuda()), Variable(y.cuda())
            ds = ds.numpy()
            numSample = numSample + y.size(0)

            y_pred = self.model(X, ds)
            _, pred = torch.max(y_pred.data, 1)
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
                current_acc = 100.0 * running_corrects / numSample
                print("Batch {}, Test ACC:{:.4f}%".format(
                    batch, 100.0 * running_corrects / numSample))
                wandb.log({"batch_test_acc": current_acc, "batch": batch})  # Log to W&B

        epoch_acc = 100.0 * running_corrects.item() / numSample
        print("{} Acc:{:.4f}%".format('test', epoch_acc))
        wandb.log({"test_accuracy": epoch_acc})  # Log to W&B
        # 打印每个数据集的准确度
        for i in range(3):
            if dataset_totals[i] > 0:
                dataset_acc = 100.0 * dataset_corrects[i] / dataset_totals[i]
                print("Dataset {} Acc: {:.4f}%".format(i + 1, dataset_acc))
                # Log to W&B
                wandb.log({f"Dataset_{i+1}_accuracy": dataset_acc})

        time_end = time.time() - time_open
        print("程序运行时间:{}分钟...".format(int(time_end / 60)))

        return epoch_acc, v_idx, v_pred, v_label


def main():
    exgan = ExGAN()
    exgan.zero_shot_test()

if __name__ == "__main__":
    main()
