import os
import torch
import torchvision.transforms as trans
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from transformers import AutoModel, CLIPImageProcessor
from PIL import Image

def image_convern(img):
    image = img.convert('RGB')
    return image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(
    'OpenGVLab/InternViT-6B-224px',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).to(device).eval()

image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-6B-224px')
student_model = torch.nn.Sequential(torch.nn.LayerNorm(3200),
    torch.nn.Linear(3200, 10)).to(device)

# model, preprocess = clip.load('ViT-B/16', device)
total = sum([param.nelement() for param in model.parameters()])
print("%s-Number of parameter: %.4fM" % ('ViT-B/16', total / 1e6))

# Load the dataset
root = os.path.expanduser("~/.cache")
transforms = trans.Compose(
            [trans.Resize(size=224, max_size=None, antialias=None),
             trans.CenterCrop(size=(224, 224)),
             trans.ToTensor(),
             ])
train = CIFAR100(root, download=True, train=True, transform=transforms)
test = CIFAR100(root, download=True, train=False, transform=transforms)


def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            img_list = []
            for img in images:
                ############# 将tensor转换为PIL格式 #######################
                img = img.squeeze().detach().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img = img.convert('RGB')
                # img.show()
                img_list.append(img)
            pixel_values = image_processor(images=img_list, return_tensors='pt').pixel_values
            pixel_values = pixel_values.to(torch.bfloat16).to(device)

            output = model(pixel_values)
            features = output.pooler_output.to(torch.float32).to(device)

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")

#### Accuracy = 94.300