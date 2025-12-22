import os
import glob
import time
import argparse
from statistics import mean, pstdev

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from tqdm import tqdm
import transformers
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

import clip  # OpenAI CLIP
import open_clip

from TrainingModel_pt import SentenceModelWithLinearTransformation


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def resolve_ckpt_list(spec: str):

    if not spec:
        return []

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    raw = []
    for p in parts:
        hits = glob.glob(p, recursive=True)
        if not hits:
            hits = [p]
        for h in hits:
            if os.path.isdir(h):
                raw.extend(glob.glob(os.path.join(h, "**", "*.pt"), recursive=True))
                raw.extend(glob.glob(os.path.join(h, "**", "*.pth"), recursive=True))
            else:
                raw.append(h)

    files = sorted({os.path.abspath(x) for x in raw if os.path.isfile(x)})

    def sort_key(x):
        name = os.path.basename(x).lower()
        best_rank = 0 if ('best' in name) else 1
        mtime = -os.path.getmtime(x)  # 최신 우선
        return (best_rank, mtime)

    files.sort(key=sort_key)
    return files


def safe_load_state_dict(model, ckpt_path, map_location='cpu'):
    state = torch.load(ckpt_path, map_location=map_location)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    cleaned = {}
    for k, v in state.items():
        if isinstance(k, str) and k.startswith('module.'):
            cleaned[k[len('module.'):]] = v
        else:
            cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"[WARN] Loading {ckpt_path}: missing={missing}, unexpected={unexpected}")


def compute_topk_accuracy(image_embeddings, text_embeddings, labels, k=1):
    if image_embeddings.dtype != text_embeddings.dtype:
        text_embeddings = text_embeddings.to(dtype=image_embeddings.dtype)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    logits = image_embeddings @ text_embeddings.T  # [N, C]
    topk = logits.topk(k, dim=1).indices          # [N, k]
    correct = topk.eq(labels.unsqueeze(1))        # [N, k]
    topk_acc = correct.any(dim=1).float().mean().item()
    return topk_acc


def eval_text_model_over_ckpts(make_model_fn, ckpt_list, input_ids, attention_mask,
                               image_embeddings, image_labels, topk_list, device):

    scores = {k: [] for k in topk_list}
    if len(ckpt_list) == 0:
        with torch.no_grad():
            model = make_model_fn().to(device).eval()
            t_emb = model(input_ids, attention_mask).float()
            for k in topk_list:
                acc = compute_topk_accuracy(image_embeddings, t_emb, labels=image_labels, k=k)
                scores[k].append(acc)
    else:
        with torch.no_grad():
            for ckpt in ckpt_list:
                model = make_model_fn().to(device).eval()
                safe_load_state_dict(model, ckpt, map_location=device)
                t_emb = model(input_ids, attention_mask).float()
                for k in topk_list:
                    acc = compute_topk_accuracy(image_embeddings, t_emb, labels=image_labels, k=k)
                    scores[k].append(acc)

    stats = {
        k: {
            "mean": mean(scores[k]),
            "std": (pstdev(scores[k]) if len(scores[k]) > 1 else 0.0),
            "runs": scores[k],
        }
        for k in topk_list
    }
    return stats


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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-100 with CLIP/MCLIP/ToMCLIP variants (multi-ckpt avg)")
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number (default: 0)')
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/32', help='CLIP model name (default: ViT-B/32)')

    parser.add_argument('--mclip_model_path', type=str, required=True, help='Path(s) to MCLIP checkpoint(s)')
    parser.add_argument('--tomclip_dm_model_path', type=str, required=True, help='Path(s) to ToMCLIP_dm checkpoint(s)')
    parser.add_argument('--tomclip_ta_model_path', type=str, required=True, help='Path(s) to ToMCLIP_ta checkpoint(s)')
    parser.add_argument('--tomclip_model_path', type=str, required=True, help='Path(s) to ToMCLIP (base) checkpoint(s)')

    parser.add_argument('--labels_csv', type=str, default='cifar100_multilingual_labels.csv',
                        help='CSV of CIFAR-100 class names by language')
    parser.add_argument('--output_csv', type=str, default='cifar100_multilingual_recall.csv',
                        help='Path to save the MEAN/STD CSV')
    parser.add_argument('--log', type=str2bool, default=True, help='Also write a .log summary next to output CSV')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print("Loading CLIP...")
    if args.clip_model_name == 'ViT-B-16-plus-240':
        CLIP_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32",device=device)
    else:
        CLIP_model, _ = clip.load(args.clip_model_name, device=device)
    CLIP_model.eval()

    print("Loading tokenizer...")
    hf_model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'  
    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    def new_sentence_model():
        return SentenceModelWithLinearTransformation('xlm-roberta-large', target_embedding_dim=512)

    print("Resolving checkpoints...")
    mclip_ckpts       = resolve_ckpt_list(args.mclip_model_path)
    tomclip_dm_ckpts  = resolve_ckpt_list(args.tomclip_dm_model_path)
    tomclip_ta_ckpts  = resolve_ckpt_list(args.tomclip_ta_model_path)
    tomclip_base_ckpts= resolve_ckpt_list(args.tomclip_model_path)

    print(f"MCLIP ckpts       : {len(mclip_ckpts)} found")
    print(f"ToMCLIP_dm ckpts  : {len(tomclip_dm_ckpts)} found")
    print(f"ToMCLIP_ta ckpts  : {len(tomclip_ta_ckpts)} found")
    print(f"ToMCLIP (base)    : {len(tomclip_base_ckpts)} found")

    if not args.clip_model_name == 'ViT-B-16-plus-240':
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
        ])

    print("Preparing CIFAR-100 test set...")
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
    test_loader = DataLoader(cifar100, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    print("Computing image embeddings with CLIP...")
    image_embeddings = []
    image_labels = []
    with torch.no_grad():
        for images, batch_labels in tqdm(test_loader):
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            feats = CLIP_model.encode_image(images).float()
            feats = F.normalize(feats, dim=-1)
            image_embeddings.append(feats)
            image_labels.append(batch_labels)
    image_embeddings = torch.cat(image_embeddings, dim=0)  # [N, 512]
    image_labels = torch.cat(image_labels, dim=0)          # [N]

    print(f"Reading labels from: {args.labels_csv}")
    cifar100_labels = pd.read_csv(args.labels_csv, encoding='utf-8')
    languages = list(cifar100_labels.columns)

    topk_list = [1, 5, 10]
    rows = []   
    runs_rows = [] 

    for lang in languages:
        if lang not in prompt_templates:
            print(f"[WARN] Language '{lang}' not in prompt_templates. Skipping.")
            continue

        print(f"\nEvaluating language: {lang}")
        class_names = cifar100_labels[lang].tolist() 
        texts = [prompt_templates[lang].format(name) for name in class_names]

        clip_tokens = clip.tokenize(texts).to(device)
        hf_inputs = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        input_ids = hf_inputs['input_ids'].to(device)
        attention_mask = hf_inputs['attention_mask'].to(device)

        with torch.no_grad():
            clip_text_emb = CLIP_model.encode_text(clip_tokens).float()
        for k in topk_list:
            acc = compute_topk_accuracy(image_embeddings, clip_text_emb, labels=image_labels, k=k)
            rows.append({
                "language": lang, "model": "CLIP", "k": k,
                "n_ckpts": 1, "accuracy_mean": round(acc * 100.0, 4), "accuracy_std": 0.0
            })
            runs_rows.append({
                "language": lang, "model": "CLIP", "k": k, "run": 1,
                "accuracy": round(acc * 100.0, 4)
            })
            print(f"Top-{k} {'CLIP':12s} : {acc*100:.2f}%")

        cfgs = [
            ("MCLIP",      mclip_ckpts),
            ("ToMCLIP_dm", tomclip_dm_ckpts),
            ("ToMCLIP_ta", tomclip_ta_ckpts),
            ("ToMCLIP",    tomclip_base_ckpts),
        ]

        for model_name, ckpt_list in cfgs:
            print(f"Evaluating {model_name} with {len(ckpt_list)} ckpt(s)...")
            if args.clip_model_name == 'ViT-B-16-plus-240':
                emb_dim = 640
            else:
                emb_dim = 512
            stats = eval_text_model_over_ckpts(
                make_model_fn=lambda: SentenceModelWithLinearTransformation('xlm-roberta-large', target_embedding_dim=emb_dim),
                ckpt_list=ckpt_list,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_embeddings=image_embeddings,
                image_labels=image_labels,
                topk_list=topk_list,
                device=device
            )

            for k in topk_list:
                m = stats[k]["mean"]; s = stats[k]["std"]; runs = stats[k]["runs"]
                rows.append({
                    "language": lang, "model": model_name, "k": k,
                    "n_ckpts": max(1, len(ckpt_list)),
                    "accuracy_mean": round(m * 100.0, 4),
                    "accuracy_std": round(s * 100.0, 4)
                })
                for i, v in enumerate(runs, start=1):
                    runs_rows.append({
                        "language": lang, "model": model_name, "k": k,
                        "run": i, "accuracy": round(v * 100.0, 4)
                    })
                print(f"Top-{k} {model_name:12s} : {m*100:.2f}% (±{s*100:.2f}, n={max(1,len(ckpt_list))})")

    df = pd.DataFrame(rows).sort_values(by=["language", "model", "k"])
    df.to_csv(args.output_csv, index=False, encoding='utf-8')
    print(f"\nSaved per-language MEAN/STD results to: {args.output_csv}")

    runs_path = os.path.splitext(args.output_csv)[0] + "_runs.csv"
    pd.DataFrame(runs_rows).sort_values(by=["language", "model", "k", "run"]) \
        .to_csv(runs_path, index=False, encoding='utf-8')
    print(f"Saved per-run results to: {runs_path}")

    avg = (df.groupby(["model", "k"])[["accuracy_mean"]]
             .mean().reset_index().sort_values(["model", "k"]))
    avg["language"] = "AVG"
    avg["n_ckpts"] = None
    avg["accuracy_std"] = None
    avg = avg[["language", "model", "k", "n_ckpts", "accuracy_mean", "accuracy_std"]]

    avg_path = os.path.splitext(args.output_csv)[0] + "_with_avg.csv"
    pd.concat([df, avg], ignore_index=True).to_csv(avg_path, index=False, encoding='utf-8')
    print(f"Saved results with AVG rows to: {avg_path}")

    if args.log:
        log_path = os.path.splitext(args.output_csv)[0] + ".log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Evaluation @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CLIP: {args.clip_model_name}\n\n")
            f.write(f"MCLIP ckpts      : {len(mclip_ckpts)}\n")
            f.write(f"ToMCLIP_dm ckpts : {len(tomclip_dm_ckpts)}\n")
            f.write(f"ToMCLIP_ta ckpts : {len(tomclip_ta_ckpts)}\n")
            f.write(f"ToMCLIP ckpts    : {len(tomclip_base_ckpts)}\n\n")

            for k in topk_list:
                f.write(f"=== Average Top-{k} (language mean of MEANs) ===\n")
                sub = avg[avg["k"] == k]
                for _, r in sub.iterrows():
                    f.write(f"{r['model']:12s}: {r['accuracy_mean']:.2f}%\n")
                f.write("\n")
        print(f"Wrote log summary to: {log_path}")


if __name__ == '__main__':
    main()
