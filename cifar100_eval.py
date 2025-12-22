import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from TrainingModel_pt import SentenceModelWithLinearTransformation
from multilingual_clip import pt_multilingual_clip
import clip 
import argparse
import time
import transformers
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate cifar100 with custom model")
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number (default: 0)')
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/32', help='CLIP model name (default: ViT-B/32)')
    parser.add_argument('--mclip_model_path', type=str, default='M-CLIP/XLM-Roberta-Large-Vit-B-32', help='Pretrained Multilingual CLIP model name')
    parser.add_argument('--tomclip_model_path', type=str, default=None, help='Path to the trained model checkpoint')
    parser.add_argument('--log', type=str2bool, default=True, help='Whether to save log file')
    return parser.parse_args()

def compute_topk_accuracy(image_embeddings, text_embeddings, labels, k=1):
    if image_embeddings.dtype != text_embeddings.dtype:
        text_embeddings = text_embeddings.to(dtype=image_embeddings.dtype)

    text_embeddings = F.normalize(text_embeddings, dim=-1)
    logits = image_embeddings @ text_embeddings.T  # [N, C]
    topk = logits.topk(k, dim=1).indices            # [N, k]
    correct = topk.eq(labels.unsqueeze(1))          # [N, k]
    topk_acc = correct.any(dim=1).float().mean().item()
    return topk_acc

prompt_templates = {
    "English": "a photo of a {}",
    "Korean": "{}가 있는 사진",
    "Japanese": "{}の写真",
    "French": "une photo d’un(e) {}",
    "Spanish": "una foto de un(a) {}",
    "Russian": "фото {}",
    "Italian": "una foto di un(a) {}",
    "Chinese": "一张{}的照片",
    "Vietnamese": "một bức ảnh về {}",
    "Danish": "et billede af en {}",
    "Polish": "zdjęcie {}",
    "Turkish": "{} fotoğrafı",
    "German": "ein Foto von einem/einer {}"
}

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    CLIP_model, preprocess = clip.load('ViT-B/32', device)

    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    MCLIP_model = SentenceModelWithLinearTransformation('xlm-roberta-large', target_embedding_dim=512)
    MCLIP_model.load_state_dict(torch.load(args.mclip_model_path, map_location=device))

    ToMCLIP_model = SentenceModelWithLinearTransformation('xlm-roberta-large', target_embedding_dim=512)
    ToMCLIP_model.load_state_dict(torch.load(args.tomclip_model_path, map_location=device))

    CLIP_model.eval()
    MCLIP_model.to(device).eval()
    ToMCLIP_model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])
    ])

    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)

    test_loader = DataLoader(cifar100, batch_size=64, shuffle=False)

    image_embeddings = []
    image_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            features = CLIP_model.encode_image(images)
            features = torch.nn.functional.normalize(features, dim=-1)
            image_embeddings.append(features)
            image_labels.append(labels)

    # Concatenate
    image_embeddings = torch.cat(image_embeddings, dim=0)
    image_labels = torch.cat(image_labels, dim=0)

    cifar100_labels = pd.read_csv("cifar100_multilingual_labels.csv", encoding='utf-8') 
    languages = cifar100_labels.keys().tolist()  

    recalls = {}

    for lang in languages:
        print(f"Evaluating XTD10 language: {lang}")
        labels = cifar100_labels[lang].tolist()
        texts = [prompt_templates[lang].format(label) for label in labels]

        # Tokenization
        text_cliptoken = clip.tokenize(texts).to(device)
        inputs = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        input_ids = inputs['input_ids'].squeeze(0).to(device)
        attention_mask = inputs['attention_mask'].squeeze(0).to(device)

        # Inference
        with torch.no_grad():
            CLIP_outputs = CLIP_model.encode_text(text_cliptoken).float()
            M_outputs = MCLIP_model(input_ids, attention_mask).float()
            ToM_outputs = ToMCLIP_model(input_ids, attention_mask).float()

        recalls[lang] = {}

        # Evaluate at multiple top-k values
        topk_list = [1, 5, 10]
        for k in topk_list:
            clip_acc = compute_topk_accuracy(image_embeddings, CLIP_outputs, labels=image_labels, k=k)
            mclip_acc = compute_topk_accuracy(image_embeddings, M_outputs, labels=image_labels, k=k)
            tomclip_acc = compute_topk_accuracy(image_embeddings, ToM_outputs, labels=image_labels, k=k)

            recalls[lang][f"clip@{k}"] = clip_acc * 100
            recalls[lang][f"mclip@{k}"] = mclip_acc * 100
            recalls[lang][f"tomclip@{k}"] = tomclip_acc * 100

            print(f"\nTop-{k} Accuracy ({lang}):")
            print(f"CLIP:     {clip_acc:.4f}")
            print(f"MCLIP:    {mclip_acc:.4f}")
            print(f"ToMCLIP:  {tomclip_acc:.4f}")

    if args.log and args.tomclip_model_path:
        log_path = os.path.join('/',*args.tomclip_model_path.split('/')[:-1], f'evaluate_cifar100_Recall.log')
        with open(log_path, 'w') as f:
            f.write(f"Evaluation is conducted at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model path: {args.tomclip_model_path}\n")
            f.write(f"CLIP model: {args.clip_model_name}\n")
            f.write(f"MCLIP model: {args.mclip_model_path}\n\n")

            for lang, score_dict in recalls.items():
                f.write(f"[{lang}]\n")
                for k_tag, val in score_dict.items():
                    f.write(f"{k_tag}: {val:.2f}%\n")
                f.write("\n")

            k_tags = list(next(iter(recalls.values())).keys()) 
            f.write("=== Average by k_tag ===\n")
            for k_tag in k_tags:
                values = [score_dict[k_tag] for score_dict in recalls.values()]
                avg_score = sum(values) / len(values)
                f.write(f"{k_tag}: {avg_score:.2f}%\n")

if __name__ == '__main__':
    main()

