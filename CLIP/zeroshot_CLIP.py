import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm
import os
import clip  # 确保导入clip模块
from datasetstest import get_data  # 从你的自定义模块导入

# Configuration: Change 'CLIP-16' to 'CLIP-14' to switch models
MODEL_NAME = 'CLIP-16'  # 修改这里来切换模型

# Mapping model names to CLIP model identifiers
model_settings = {
    "CLIP-16": "ViT-B/16",
    "CLIP-14": "ViT-L/14"
}

class ClipModel(nn.Module):
    def __init__(self, pretrained_model):
        super(ClipModel, self).__init__()
        self.model, _ = clip.load(pretrained_model, device=device)  
        self.linear = nn.Linear(512, 18)  # CLIP-16的输出维度是512，假设有18个类别

    def forward(self, image):
        features = self.model.encode_image(image)
        logits = self.linear(features)
        return logits

def evaluate_model(model, dataloader):
    model.eval()
    dataset_correct = [0, 0, 0]
    dataset_total = [0, 0, 0]
    results = []
    with torch.no_grad():
        for images, labels, dataset_ids in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probabilities = nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            for i in range(len(labels)):
                index = dataset_ids[i] - 1
                dataset_total[index] += 1
                correct = (predicted[i] == labels[i]).item()
                dataset_correct[index] += correct
                results.append({
                    "image_path": dataloader.dataset.root + dataloader.dataset.files[dataset_ids[i]],
                    "image_name": dataloader.dataset.files[dataset_ids[i]],
                    "label": labels[i].item(),
                    "prediction": predicted[i].item(),
                    "dataset_id": index
                })
    dataset_accuracy = [correct / total for correct, total in zip(dataset_correct, dataset_total)]
    overall_accuracy = sum(dataset_correct) / sum(dataset_total)
    return overall_accuracy, dataset_accuracy, results

# Load the data
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
_, test_loader = get_data(preprocess, batch_size=32)

# Initialize and evaluate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = model_settings[MODEL_NAME]
model = ClipModel(clip_model_name).to(device)
model = nn.DataParallel(model)  # Enable multi-GPU
overall_acc, dataset_acc, inference_results = evaluate_model(model, test_loader)

# Create a directory for saving results
results_dir = 'zeroshot_results'
os.makedirs(results_dir, exist_ok=True)

# Save the results to a JSONL file in the specified directory
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = os.path.join(results_dir, f'inference_results_{MODEL_NAME}_{current_time}.jsonl')
with open(filename, 'w') as f:
    for result in inference_results:
        f.write(json.dumps(result) + '\n')  # 确保所有字段都是可序列化的

print(f"Overall Model Accuracy: {overall_acc * 100:.2f}%")
for i, acc in enumerate(dataset_acc, 1):
    print(f"Dataset {i} Accuracy: {acc * 100:.2f}%")
