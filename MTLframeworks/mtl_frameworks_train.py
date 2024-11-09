import time
import datetime as datetime
from torch.autograd import Variable
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import timm
import clip
from datasets_mtl_frameworks import *  # 请确保您的数据集文件名正确
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os
from torchvision import models

# 初始化W&B
wandb.init(project="Baselines_Frameworks")
config = wandb.config

config.batch_size = 24
config.n_epochs = 10
config.learning_rate = 0.00005

torch.cuda.set_device(0)

# 多任务框架实现
class SimpleCrossStitch(nn.Module):
    def __init__(self, backbone_model, dim):
        super(SimpleCrossStitch, self).__init__()
        self.backbone = backbone_model
        self.cls1 = nn.Sequential(nn.Linear(dim, 10)).cuda()
        self.cls2 = nn.Sequential(nn.Linear(dim, 6)).cuda()
        self.cls3 = nn.Sequential(nn.Linear(dim, 2)).cuda()
        self.cross_stitch = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]), requires_grad=True)
        self.criterion_cls = nn.CrossEntropyLoss().cuda()

    def forward(self, x, ds, label):
        x = x.cuda()
        # print("Input shape to backbone:", x.shape)  # 打印输入形状
        feature = self.backbone(x).float()
        # print("Feature shape after backbone:", feature.shape)  # 打印特征形状

        shared_feature = feature * self.cross_stitch[0]
        task_feature1 = feature * self.cross_stitch[1]
        task_feature2 = feature * self.cross_stitch[2]
        loss = 0
        predictions = []
        for i, d in enumerate(ds):
            y = label[i].unsqueeze(0).cuda()
            
            # 计算各个任务特征
            task1_output = self.cls1(shared_feature + task_feature1)
            task2_output = self.cls2(shared_feature + task_feature2)
            task3_output = self.cls3(feature)
            
            # 根据 d 选择输出层
            out = [task1_output, task2_output, task3_output][d][i].unsqueeze(0)
            
            _, pred = torch.max(out.data, 1)
            loss += self.criterion_cls(out, y)
            predictions.append(pred)
        predictions = torch.cat(predictions, dim=0)
        return predictions, loss

class SimpleMTAN(nn.Module):
    def __init__(self, backbone_model, dim):
        super(SimpleMTAN, self).__init__()
        self.backbone = backbone_model
        self.attention1 = nn.Parameter(torch.rand(dim)).cuda()
        self.attention2 = nn.Parameter(torch.rand(dim)).cuda()
        self.attention3 = nn.Parameter(torch.rand(dim)).cuda()
        self.cls1 = nn.Sequential(nn.Linear(dim, 10)).cuda()
        self.cls2 = nn.Sequential(nn.Linear(dim, 6)).cuda()
        self.cls3 = nn.Sequential(nn.Linear(dim, 2)).cuda()
        self.criterion_cls = nn.CrossEntropyLoss().cuda()

    def forward(self, x, ds, label):
        x = x.cuda()
        feature = self.backbone(x).float()
        loss = 0
        predictions = []
        for i, d in enumerate(ds):
            y = label[i].unsqueeze(0).cuda()
            task_feature = feature[i] * [self.attention1, self.attention2, self.attention3][d]
            out = [self.cls1, self.cls2, self.cls3][d](task_feature.unsqueeze(0))
            _, pred = torch.max(out.data, 1)
            loss += self.criterion_cls(out, y)
            predictions.append(pred)
        predictions = torch.cat(predictions, dim=0)
        return predictions, loss

class SimpleMoE(nn.Module):
    def __init__(self, backbone_model, dim):
        super(SimpleMoE, self).__init__()
        self.backbone = backbone_model
        self.expert1 = nn.Linear(dim, 10).cuda()
        self.expert2 = nn.Linear(dim, 6).cuda()
        self.expert3 = nn.Linear(dim, 2).cuda()
        self.criterion_cls = nn.CrossEntropyLoss().cuda()

    def forward(self, x, ds, label):
        x = x.cuda()
        feature = self.backbone(x).float()
        loss = 0
        predictions = []
        for i, d in enumerate(ds):
            y = label[i].unsqueeze(0).cuda()
            out = [self.expert1, self.expert2, self.expert3][d](feature[i].unsqueeze(0))
            _, pred = torch.max(out.data, 1)
            loss += self.criterion_cls(out, y)
            predictions.append(pred)
        predictions = torch.cat(predictions, dim=0)
        return predictions, loss

def initialize_model(model_name, framework_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 通用预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if model_name == "CLIP-16":
        clip_model, preprocess = clip.load('ViT-B/16', device)
        dim = 512
        backbone_model = clip_model
    elif model_name == "CLIP-14":
        clip_model, preprocess = clip.load('ViT-L/14', device)
        dim = 768
        backbone_model = clip_model
    elif model_name == "ResNet50":
        backbone_model = timm.create_model('resnet50', pretrained=True).cuda()
        dim = backbone_model.get_classifier().in_features
        backbone_model.reset_classifier(0)
        
        # 更新：使用AdaptiveAvgPool2d将输出压缩为feature_dim，再进行展平
        backbone_model = nn.Sequential(
            *list(backbone_model.children())[:-2],  # 移除最后的 SelectAdaptivePool2d 和 Identity 层
            nn.AdaptiveAvgPool2d((1, 1)),          # 添加 AdaptiveAvgPool2d 使输出符合 [batch_size, feature_dim]
            nn.Flatten(start_dim=1)                # 展平为 [batch_size, feature_dim]
        )
    elif model_name == "VGG16":
        backbone_model = models.vgg16(pretrained=True).cuda()
        dim = backbone_model.classifier[-1].in_features
        backbone_model.classifier = backbone_model.classifier[:-1]
        backbone_model = nn.Sequential(backbone_model, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    elif model_name == "EfficientNet-B0":
        backbone_model = timm.create_model('efficientnet_b0', pretrained=True).cuda()
        dim = backbone_model.classifier.in_features
        backbone_model.reset_classifier(0)
        backbone_model = nn.Sequential(backbone_model, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    elif model_name == "DenseNet121":
        backbone_model = models.densenet121(pretrained=True).cuda()
        dim = backbone_model.classifier.in_features
        backbone_model.classifier = nn.Identity()
        backbone_model = nn.Sequential(backbone_model, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    elif model_name == "ConvNeXt-Base":
        backbone_model = timm.create_model('convnext_base', pretrained=True).cuda()
        dim = backbone_model.head.fc.in_features
        backbone_model.head.fc = nn.Identity()
        backbone_model = nn.Sequential(backbone_model, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    elif model_name == "ViT-Base":
        backbone_model = timm.create_model('vit_base_patch16_224', pretrained=True).cuda()
        dim = backbone_model.head.in_features
        backbone_model.head = nn.Identity()
    elif model_name == "MobileNetV3-Large":
        backbone_model = models.mobilenet_v3_large(pretrained=True).cuda()
        dim = backbone_model.classifier[-1].in_features
        backbone_model.classifier = nn.Identity()
        backbone_model = nn.Sequential(backbone_model, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    else:
        raise ValueError("Unsupported model name!")

    # 打印逐层输出维度以检测问题
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    print(f"Running forward pass with sample input shape {sample_input.shape} on model {model_name} ...")
    x = sample_input
    try:
        for i, layer in enumerate(backbone_model):
            x = layer(x)
            print(f"Layer {i}: {layer.__class__.__name__}, Output shape: {x.shape}")
    except Exception as e:
        print(f"Error at Layer {i}: {layer.__class__.__name__}, with shape {x.shape}")
        raise e

    # 初始化多任务框架
    if framework_name == "SimpleCrossStitch":
        model = SimpleCrossStitch(backbone_model, dim).cuda()
    elif framework_name == "SimpleMTAN":
        model = SimpleMTAN(backbone_model, dim).cuda()
    elif framework_name == "SimpleMoE":
        model = SimpleMoE(backbone_model, dim).cuda()
    else:
        raise ValueError("Unsupported framework name!")
    
    return model, preprocess




class ExGAN():
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess
        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs
        self.lr = config.learning_rate
        self.criterion_cls = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.dataloader, self.test_dataloader = get_data(self.preprocess, self.batch_size)
        self.log_write = open(f"./results/log_results_in.txt", "w")

    def train(self):
        print(f"Starting training with {self.model}")
        best_accuracy = 0
        average_accuracy = 0
        num_test = 0
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}")
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            num_sample = 0
            for batch, (imgA, label, ds) in enumerate(self.dataloader, 1):
                X, y = imgA, label
                X, y = Variable(X.cuda()), Variable(y.cuda())
                ds = ds.numpy()
                num_sample += y.size(0)
                pred, loss = self.model(X, ds, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(pred == y.data)

                if batch % 100 == 0:
                    print(f"Batch {batch} - Loss: {running_loss / batch:.4f} - Accuracy: {100.0 * running_corrects / num_sample:.4f}%")
                    wandb.log({"train_loss": running_loss / batch, "train_accuracy": 100.0 * running_corrects / num_sample})

            epoch_acc = 100.0 * running_corrects / num_sample
            print(f"Epoch {epoch + 1} - Accuracy: {epoch_acc:.4f}%")
            wandb.log({"epoch_accuracy": epoch_acc})
            
            # 仅使用 epoch_acc 而不是整个元组
            test_acc, _, _, _ = self.test(self.test_dataloader)
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(self.model.state_dict(), f"saved_models/best_model.pth")
            average_accuracy += test_acc
            num_test += 1

        average_accuracy /= num_test
        print(f'Best Test Accuracy: {best_accuracy:.4f}%, Average Test Accuracy: {average_accuracy:.4f}%')
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


# 主函数
def main():
    # model_names = ['ResNet50', 'EfficientNet-B0', 'VGG16', 'DenseNet121', 'ConvNeXt-Base', 'ViT-Base', 'MobileNetV3-Large', 'CLIP-16']
    model_names = ['EfficientNet-B0', 'VGG16', 'DenseNet121', 'ViT-Base', 'MobileNetV3-Large', 'CLIP-16']  
    frameworks = ['SimpleCrossStitch', 'SimpleMTAN', 'SimpleMoE']  

    for model_name in model_names:
        for framework_name in frameworks:
            print(f"Training with model: {model_name} and framework: {framework_name}")
            wandb.init(project="Baselines_Frameworks", name=f"{model_name}_{framework_name}_run", reinit=True)
            model, preprocess = initialize_model(model_name, framework_name)
            exgan = ExGAN(model, preprocess)
            exgan.train()
            wandb.finish()

if __name__ == "__main__":
    main()
