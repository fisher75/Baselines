import os
import torch
import json
from PIL import Image
import clip
import wandb
from tqdm import tqdm

# 初始化W&B
wandb.init(project="CLIP_ZeroShot")

# 定义数据集路径和类别描述
dataset_path = "/home/users/ntu/chih0001/scratch/data/mixed/test"
datasets = {
    "ds1": ["Normal Driving", "Drinking", "Phoning Left", "Phoning Right", "Texting Left", "Texting Right", "Touching Hairs & Makeup", "Adjusting Glasses", "Reaching Behind", "Dropping"],
    "ds2": ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"],
    "ds3": ["Drowsy", "Non Drowsy"]
}

# 可选的CLIP模型
clip_models = ['ViT-B/16', 'ViT-B/32', 'ViT-L/14']

# 函数：加载和预处理图像
def load_and_preprocess_image(image_path, preprocess, device):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

# 函数：进行零样本分类
def classify_images(model, preprocess, device, model_name):
    # 为每个数据集创建文本标签并tokenize
    text_inputs = []
    label_texts = []
    for dataset, descriptions in datasets.items():
        texts = [f"a photo of {desc}" for desc in descriptions]
        label_texts.extend(descriptions)
        tokenized = torch.cat([clip.tokenize(text).to(device) for text in texts])
        text_inputs.append(tokenized)

    text_inputs = torch.cat(text_inputs, dim=0)

    # 遍历测试集图片
    results = []
    correct_count = {key: 0 for key in datasets}
    total_count = {key: 0 for key in datasets}
    for image_name in tqdm(os.listdir(dataset_path), desc=f"Processing images with {model_name}"):
        image_path = os.path.join(dataset_path, image_name)
        image_input = load_and_preprocess_image(image_path, preprocess, device)

        # 计算特征
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # 计算相似度并选择最高的预测
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        # 记录结果
        predicted_label = label_texts[indices[0].item()]
        actual_label_index = int(image_name.split('_')[1][3])  # Extract label index from filename
        actual_label = datasets[image_name[:3]][actual_label_index]
        if predicted_label == actual_label:
            correct_count[image_name[:3]] += 1
        total_count[image_name[:3]] += 1
        results.append({
            "image_path": image_path,
            "image_name": image_name,
            "label": actual_label,
            "prediction": predicted_label,
            "dataset_id": image_name[:3],
            "model": model_name
        })

    # Calculate accuracy for each dataset
    accuracies = {k: 100 * correct_count[k] / total_count[k] for k in datasets}
    return results, accuracies, correct_count, total_count

# 主函数
def main():
    results_path = "/home/users/ntu/chih0001/scratch/model/baselines/CLIP/zeroshot_results"
    os.makedirs(results_path, exist_ok=True)

    for model_name in clip_models:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model_name, device)
        print(f"Processing with model: {model_name}")
        results, accuracies, correct_count, total_count = classify_images(model, preprocess, device, model_name)

        # 保存结果到JSONL文件和TXT文件
        with open(os.path.join(results_path, f"results_{model_name.replace('/', '-')}.jsonl"), 'w') as jsonl_file, \
             open(os.path.join(results_path, f"accuracy_{model_name.replace('/', '-')}.txt"), 'w') as txt_file:
            for result in results:
                jsonl_file.write(json.dumps(result) + '\n')
            txt_file.write(f"Accuracies for {model_name}:\n")
            for dataset, acc in accuracies.items():
                txt_file.write(f"{dataset}: {acc:.2f}% ({correct_count[dataset]}/{total_count[dataset]})\n")
                print(f"Accuracy for {dataset} using {model_name}: {acc:.2f}% ({correct_count[dataset]}/{total_count[dataset]})")

    wandb.finish()

if __name__ == "__main__":
    main()
