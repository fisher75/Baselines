import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm
import os
import clip
from datasetstest import get_data  # 导入datasetstest中的get_data

MODEL_VERSION = 'ViT-B/16'  # 正确的模型名称

class ClipModel(nn.Module):
    def __init__(self, pretrained_model=MODEL_VERSION):
        super(ClipModel, self).__init__()
        self.model, self.preprocess = clip.load(pretrained_model, device=device)  # 确保使用正确的模型名称

    def forward(self, image):
        image = self.preprocess(image).unsqueeze(0).to(device)  # Preprocess and add batch dimension
        return self.model.encode_image(image)

# Initialize and evaluate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClipModel().to(device)
model = nn.DataParallel(model)  # Enable multi-GPU

# Load the data using get_data from datasetstest.py
test_loader = get_data(preprocess=model.preprocess, batch_size=32, mode='test', shuffle=False)

def evaluate_model(model, dataloader):
    model.eval()
    dataset_correct = [0, 0, 0]
    dataset_total = [0, 0, 0]
    results = []
    with torch.no_grad():
        for images, labels, dataset_ids in tqdm(dataloader, desc="Evaluating", unit="batch"):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
                index = dataset_ids[i] - 1  # Adjust index because dataset_ids start from 1
                dataset_total[index] += 1
                dataset_correct[index] += (predicted[i] == labels[i]).item()
                results.append({
                    "image_name": dataloader.dataset.files[dataset_ids[i]],
                    "label": labels[i].item(),
                    "prediction": predicted[i].item(),
                    "dataset_id": dataset_ids[i]
                })
    dataset_accuracy = [correct / total for correct, total in zip(dataset_correct, dataset_total)]
    overall_accuracy = sum(dataset_correct) / sum(dataset_total)
    return overall_accuracy, dataset_accuracy, results

overall_acc, dataset_acc, inference_results = evaluate_model(model, test_loader)

# Output results
print(f"Overall Model Accuracy: {overall_acc * 100:.2f}%")
for i, acc in enumerate(dataset_acc, 1):
    print(f"Dataset {i} Accuracy: {acc * 100:.2f}%")
