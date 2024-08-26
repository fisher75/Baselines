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

# 初始化W&B
wandb.init(project="Baselines_CLIP")
config = wandb.config

config.batch_size = 24
config.n_epochs = 10
config.learning_rate = 0.00005

torch.cuda.set_device(0)

class ClipModel(nn.Module):
    def __init__(self, clip_model, dim):
        super(ClipModel, self).__init__()
        self.classifiers = []
        self.conv = clip_model.encode_image # 共享的encoder部分
        self.cls1 = nn.Sequential(nn.Linear(dim, 10)) # 任务1的分类器
        self.classifiers.append(self.cls1)
        self.cls2 = nn.Sequential(nn.Linear(dim, 6)) # 任务2的分类器
        self.classifiers.append(self.cls2)
        self.cls3 = nn.Sequential(nn.Linear(dim, 2)) # 任务3的分类器
        self.classifiers.append(self.cls3)
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

    def forward(self, x, ds, label):
        loss = 0
        feature = self.conv(x).float()  # 确保 feature 的数据类型为 Float
        predictions = []
        for i, d in enumerate(ds):
            y = label[i].unsqueeze(0)
            out = self.classifiers[d](feature[i].unsqueeze(0))
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
        self.log_write = open("./results/log_%s_results_in.txt" % (self.model_name), "w")
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if model_name == "CLIP-16":
            clip_model, self.preprocess = clip.load('ViT-B/16', device)
            self.dim = 512
        elif model_name == "CLIP-14":
            clip_model, self.preprocess = clip.load('ViT-L/14', device)
            self.dim = 768
        self.model = ClipModel(clip_model, self.dim)
        self.model = self.model.cuda()
        total = sum([param.nelement() for param in self.model.parameters()])
        print("%s-Number of parameter: %.4fM" % (self.model_name, total / 1e6))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.Transform = self.preprocess
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
                        current_date = datetime.datetime.now().strftime("%Y-%m%d")
                        torch.save(self.model.state_dict(), "saved_models/model_{}_{}_{}_weights.pth".format(self.model_name, self.task_name, current_date))
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
    models_list = ['CLIP-16']  # Assuming CLIP-16 corresponds to 'ViT-B/16'
    for model in models_list:
        exgan = ExGAN(model)
        exgan.train()

if __name__ == "__main__":
    main()
