import time
import datetime as datetime
from torch.autograd import Variable
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import timm
import clip
from datasets_3heads import *
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os
from torchvision import models  # 使用 torchvision 加载模型


# 初始化W&B
wandb.init(project="Baselines_Backbones")
config = wandb.config

config.batch_size = 24
config.n_epochs = 10
config.learning_rate = 0.00005

torch.cuda.set_device(0)

#多个Class用多种backbone：比如Resnet/VGG

class GenericModel(nn.Module):
    def __init__(self, backbone_model, dim):
        super(GenericModel, self).__init__()
        self.backbone = backbone_model  # 不调用 .cuda()，已经在ExGAN中处理
        self.cls1 = nn.Sequential(nn.Linear(dim, 10)).cuda()  # 任务1的分类器
        self.cls2 = nn.Sequential(nn.Linear(dim, 6)).cuda()   # 任务2的分类器
        self.cls3 = nn.Sequential(nn.Linear(dim, 2)).cuda()   # 任务3的分类器
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

    def forward(self, x, ds, label):
        x = x.cuda()  # 将输入数据放到GPU上
        loss = 0
        feature = self.backbone(x).float()  # 调用整个模型进行特征提取
        predictions = []
        for i, d in enumerate(ds):
            y = label[i].unsqueeze(0).cuda()  # 将label放到GPU上
            out = [self.cls1, self.cls2, self.cls3][d](feature[i].unsqueeze(0))
            _, pred = torch.max(out.data, 1)
            loss += self.criterion_cls(out, y)
            predictions.append(pred)
        predictions = torch.cat(predictions, dim=0)
        return predictions, loss



class ExGAN():
    def __init__(self, model_name):
        super(ExGAN, self).__init__()
        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs
        self.img_height = 224
        self.img_width = 224
        self.channels = 3
        self.model_name = model_name
        self.task_name = "CLIP"
        self.lr = config.learning_rate
        self.b1 = 0.5
        self.b2 = 0.999
        self.log_write = open(f"./results/log_{self.model_name}_results_in.txt", "w")
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if model_name == "CLIP-16":
            clip_model, self.preprocess = clip.load('ViT-B/16', device)
            self.dim = 512
            self.model = GenericModel(clip_model, self.dim)
        elif model_name == "CLIP-14":
            clip_model, self.preprocess = clip.load('ViT-L/14', device)
            self.dim = 768
            self.model = GenericModel(clip_model, self.dim)
        elif model_name == "ResNet50":
            backbone_model = timm.create_model('resnet50', pretrained=True).cuda()  # 将整个模型加载到GPU
            self.dim = backbone_model.get_classifier().in_features
            backbone_model.reset_classifier(0)  # 移除 ResNet 的分类头
            self.preprocess = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model = GenericModel(backbone_model, self.dim)  # 传递整个模型
        elif model_name == "VGG16":
            backbone_model = models.vgg16(pretrained=True).cuda()  # 使用 torchvision 加载 VGG16
            self.dim = backbone_model.classifier[-1].in_features  # 获取最后一层的输入特征维度
            backbone_model.classifier = backbone_model.classifier[:-1]  # 移除 VGG 的分类头
            self.preprocess = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model = GenericModel(backbone_model, self.dim)  # 传递整个模型
        elif model_name == "EfficientNet-B0":
            backbone_model = timm.create_model('efficientnet_b0', pretrained=True).cuda()
            self.dim = backbone_model.classifier.in_features
            backbone_model.reset_classifier(0)
            self.preprocess = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model = GenericModel(backbone_model, self.dim)
        elif model_name == "DenseNet121":
            backbone_model = models.densenet121(pretrained=True).cuda()
            self.dim = backbone_model.classifier.in_features
            backbone_model.classifier = nn.Identity()  # 移除 DenseNet 的分类头
            self.preprocess = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model = GenericModel(backbone_model, self.dim)
        elif model_name == "ConvNeXt-Base":
            backbone_model = timm.create_model('convnext_base', pretrained=True).cuda()
            self.dim = backbone_model.head.fc.in_features
            backbone_model.head.fc = nn.Identity()  # 移除分类头
            self.preprocess = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model = GenericModel(backbone_model, self.dim)
        elif model_name == "ViT-Base":
            backbone_model = timm.create_model('vit_base_patch16_224', pretrained=True).cuda()
            self.dim = backbone_model.head.in_features
            backbone_model.head = nn.Identity()  # 移除分类头
            self.preprocess = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model = GenericModel(backbone_model, self.dim)
        # elif model_name == "MobileNetV3-Large":
        #     backbone_model = models.mobilenet_v3_large(pretrained=True).cuda()
        #     self.dim = backbone_model.classifier[-1].in_features
        #     backbone_model.classifier = nn.Identity()  # 移除 MobileNet 的分类头
        #     self.preprocess = transforms.Compose([
        #         transforms.Resize((self.img_height, self.img_width)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ])
        #     self.model = GenericModel(backbone_model, self.dim)
        elif model_name == "MobileNetV3-Large":
            backbone_model = models.mobilenet_v3_large(pretrained=True).cuda()
            self.dim = 960  # 设置 MobileNetV3-Large 的输出特征维度为 960
            backbone_model.classifier = nn.Identity()  # 移除 MobileNet 的分类头
            self.preprocess = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model = GenericModel(backbone_model, self.dim)
        else:
            raise ValueError("Unsupported model name!")

        self.model = self.model.cuda()
        total = sum([param.nelement() for param in self.model.parameters()])
        print(f"{self.model_name}-Number of parameter: {total / 1e6:.4f}M")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.Transform = self.preprocess
        self.dataloader, self.test_dataloader = get_data(self.Transform, self.batch_size)


    def train(self):
        print(f'Train on the {self.model_name} model')
        time_open = time.time()
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

            for batch, (imgA, label, ds) in enumerate(self.dataloader, 1):
                X, y = imgA, label
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
                    print(f"Epoch {epoch}/{self.n_epochs - 1}, Batch {batch}, Train Loss:{running_loss / batch:.4f}, Train ACC:{100.0 * running_corrects / numSample:.4f}%")

                if batch % 200 == 0:
                    testAcc, vidx, vpred, vlable = self.test(self.test_dataloader)
                    num_test += 1
                    self.log_write.write(f"{epoch}    {batch}    {testAcc}\n")
                    average_accuracy += testAcc
                    if testAcc > best_accuracy:
                        best_accuracy = testAcc
                        current_date = datetime.datetime.now().strftime("%Y-%m%d")
                        torch.save(self.model.state_dict(), f"saved_models/model_{self.model_name}_{self.task_name}_{current_date}_weights.pth")

            epoch_loss = running_loss * 16 / numSample
            epoch_acc = 100.0 * running_corrects / numSample

            print(f"train Loss:{epoch_loss:.4f} Acc:{epoch_acc:.4f}%")

            time_end = time.time() - time_open
            print(f"程序运行时间:{int(time_end / 60)}分钟...")

            testAcc, vidx, vpred, vlable = self.test(self.test_dataloader)
            num_test += 1
            self.log_write.write(f"{epoch}    {batch}    {testAcc}\n")
            average_accuracy += testAcc
            if testAcc > best_accuracy:
                best_accuracy = testAcc
                torch.save(self.model.state_dict(), f"saved_models/model_{self.model_name}_{self.task_name}_weights.pth")

        average_accuracy /= num_test
        print(f'The best accuracy is: {best_accuracy}\n')
        print(f'The average accuracy is: {average_accuracy}\n')
        self.log_write.write(f'The best accuracy is: {best_accuracy}\n')
        self.log_write.write(f'The average accuracy is: {average_accuracy}\n')
        self.log_write.close()

    def test(self, dataloader):
        time_open = time.time()
        running_corrects = 0
        numSample = 0
        s_idx = 0
        v_idx, v_pred, v_label, v_ds = [], [], [], []

        dataset_corrects = [0, 0, 0]
        dataset_totals = [0, 0, 0]

        for batch, (imgA, label, ds) in enumerate(dataloader, 1):
            X, y = imgA, label
            X, y = Variable(X.cuda()), Variable(y.cuda())
            ds = ds.numpy()
            numSample += y.size(0)

            pred, _ = self.model(X, ds, y)
            running_corrects += torch.sum(pred == y.data)

            for k in range(len(label)):
                dataset_index = ds[k]
                dataset_totals[dataset_index] += 1
                if pred[k] == label[k]:
                    dataset_corrects[dataset_index] += 1

                v_idx.append(s_idx)
                v_pred.append(pred[k].item())
                v_label.append(label[k].item())
                v_ds.append(ds[k])
                s_idx += 1

            if batch % 100 == 0:
                current_acc = 100.0 * running_corrects / numSample
                print("Batch {}, Test ACC:{:.4f}%".format(
                    batch, current_acc))
                wandb.log({"batch_test_acc": current_acc, "batch": batch})

        epoch_acc = 100.0 * running_corrects.item() / numSample
        print("{} Acc:{:.4f}%".format('test', epoch_acc))
        wandb.log({"test_accuracy": epoch_acc})

        for i in range(3):
            if dataset_totals[i] > 0:
                dataset_acc = 100.0 * dataset_corrects[i] / dataset_totals[i]
                print("Dataset {} Acc: {:.4f}%".format(i + 1, dataset_acc))
                wandb.log({f"Dataset_{i+1}_accuracy": dataset_acc})

        time_end = time.time() - time_open
        print("程序运行时间:{}分钟...".format(int(time_end / 60)))

        # 保存预测结果
        results = pd.DataFrame({
            'index': v_idx,
            'prediction': v_pred,
            'label': v_label,
            'dataset': v_ds
        })
        results_dir = './results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        current_date = datetime.datetime.now().strftime("%Y-%m%d")
        results_path = os.path.join(results_dir, f"CLIP_{current_date}_results.csv")
        results.to_csv(results_path, index=False)

        # 计算并记录指标
        for i in range(3):
            ds_mask = (results['dataset'] == i)
            ds_preds = results[ds_mask]['prediction']
            ds_labels = results[ds_mask]['label']

            if len(ds_labels) > 0:
                accuracy = accuracy_score(ds_labels, ds_preds)
                precision = precision_score(ds_labels, ds_preds, average='weighted', zero_division=0)
                recall = recall_score(ds_labels, ds_preds, average='weighted', zero_division=0)
                f1 = f1_score(ds_labels, ds_preds, average='weighted', zero_division=0)

                print(f"Dataset {i+1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
                wandb.log({
                    f"Dataset_{i+1}_accuracy": accuracy,
                    f"Dataset_{i+1}_precision": precision,
                    f"Dataset_{i+1}_recall": recall,
                    f"Dataset_{i+1}_f1_score": f1
                })

        return epoch_acc, v_idx, v_pred, v_label


def main():
    # models_list = ['CLIP-16', 'ResNet50', 'VGG16', 'EfficientNet-B0', 'DenseNet121', 'ConvNeXt-Base', 'ViT-Base', 'MobileNetV3-Large'] # 添加新的模型
    # models_list = ['VGG16']  # 添加新的模型
    models_list = ['MobileNetV3-Large'] # 列表里的会循环记录在wandb里边
    for model in models_list:
        wandb.init(project="Baselines_Backbones", name=f"{model}_run", reinit=True)  # 每个模型的独立 run
        exgan = ExGAN(model)
        exgan.train()
        wandb.finish()  # 每个模型训练后结束运行


if __name__ == "__main__":
    main()
