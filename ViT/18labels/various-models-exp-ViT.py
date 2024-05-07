import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision.transforms as trans
from datasets import get_data
import timm
import wandb
from tqdm import tqdm

torch.cuda.set_device(0)

class ViTModel(nn.Module):
    def __init__(self, vit_model, dim):
        super(ViTModel, self).__init__()
        self.classifiers = []
        self.conv = vit_model
        # 三个数据集分别对应0, 1, 2的索引
        self.classifiers.append(nn.Sequential(nn.Linear(dim, 10)))  # for ds1
        self.classifiers.append(nn.Sequential(nn.Linear(dim, 6)))   # for ds2
        self.classifiers.append(nn.Sequential(nn.Linear(dim, 2)))   # for ds3

    def forward(self, x, ds):
        feature = self.conv(x).type(torch.float32)
        # 确保ds是整数标量
        out = self.classifiers[int(ds)](feature)
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
        self.lr = 0.00005
        self.b1 = 0.5
        self.b2 = 0.999
        self.log_write = open("./results/log_%s_results_in.txt" % (self.model_name), "w")
        self.img_shape = (self.channels, self.img_height, self.img_width)
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = nn.L1Loss().cuda()
        self.criterion_l2 = nn.MSELoss().cuda()
        self.criterion_cls = nn.CrossEntropyLoss().cuda()

        ViT_model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=5)
        ViT_model.head = nn.Sequential()
        self.dim = 192
        self.model = ViTModel(ViT_model, self.dim).cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.Transform = trans.Compose([
            transforms.Resize(self.img_height),
            transforms.CenterCrop(self.img_height),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataloader, self.test_dataloader = get_data(self.Transform, self.batch_size)

    def train(self):
        wandb.init(project="Baselines_ViT")
        wandb.watch(self.model, log_freq=100)
        for epoch in range(self.n_epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = [0, 0, 0]
            total_samples = [0, 0, 0]
            loop = tqdm(self.dataloader, leave=True)

            for batch, (imgA, labelA, ds) in enumerate(loop):
                X, y = Variable(imgA.cuda()), Variable(labelA.cuda())
                # 确保ds是整数标量
                ds = ds.numpy()
                ds = [int(d) for d in ds]

                self.optimizer.zero_grad()
                y_pred = self.model(X, ds)
                loss = self.criterion_cls(y_pred, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, pred = torch.max(y_pred.data, 1)
                for i in range(3):
                    mask = (torch.tensor(ds) == i)
                    running_corrects[i] += torch.sum(pred[mask] == y[mask].data)
                    total_samples[i] += torch.sum(mask)

                loop.set_description(f"Epoch [{epoch+1}/{self.n_epochs}]")
                loop.set_postfix(loss=running_loss / (batch + 1))

            for i in range(3):
                acc = 100.0 * running_corrects[i] / total_samples[i]
                print(f"Dataset {i+1} Acc: {acc:.2f}%")
                wandb.log({f"Accuracy_ds{i+1}": acc})

            wandb.log({"loss": running_loss / len(self.dataloader)})

        wandb.finish()

def main():
    exgan = ExGAN()
    exgan.train()

if __name__ == "__main__":
    main()
