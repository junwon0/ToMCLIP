#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_xflickrco_v2.py
- Recursively loads annotation files (jsonl/json/csv/tsv) under --ann_path
- Filters by --split (default: test) so it ignores train_*.jsonl by default
- Infers language from directory name (e.g., .../annotations/de/test.jsonl) if "language" field is missing
- Computes IR/TR Recall@{1,5,10} with optional bootstrap CIs

Usage example:
  python evaluate_xflickrco_v2.py \
    --gpu 0 \
    --clip_model_name ViT-B/32 \
    --mclip_model_path /checkpoints/mclip.pt \
    --tomclip_model_path /checkpoints/tomclip.pt \
    --coco_root /data/COCO/val2014 \
    --flickr_root /data/flickr30k/flickr30k-images \
    --ann_path /data/iglue/datasets/xFlickrCO/annotations \
    --split test \
    --bootstrap 10000 \
    --out xflickrco_results.csv
"""
import os, re, json, glob, math, time, random, argparse
from typing import Dict, List, Tuple
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

import transformers
import clip  # OpenAI CLIP
from TrainingModel_pt import SentenceModelWithLinearTransformation

# --------------------- utils ---------------------
def set_deterministic(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)

def human_pct(x: float) -> str:
    return 'n/a' if math.isnan(x) else f'{x*100:.2f}%'

def bootstrap_ci(flags: np.ndarray, conf: float=0.95, n: int=10000, seed: int=123):
    if flags.size == 0: return (float('nan'), float('nan'))
    rng = np.random.RandomState(seed)
    N = len(flags); stats = np.empty(n, dtype=np.float32)
    for i in range(n):
        idx = rng.randint(0, N, size=N)
        stats[i] = flags[idx].mean()
    lo = np.percentile(stats, (1-conf)/2*100); hi = np.percentile(stats, (1+conf)/2*100)
    return float(lo), float(hi)

def compute_recalls(sim: torch.Tensor, ks=(1,5,10), bootstrap: int=0):
    out = {}
    N = sim.size(0); gt = torch.arange(N, device=sim.device).unsqueeze(1)
    for k in ks:
        topk = sim.topk(k, dim=1).indices
        flags = (topk == gt).any(dim=1).float()
        acc = float(flags.mean().item())
        lo, hi = (float('nan'), float('nan'))
        if bootstrap > 0:
            lo, hi = bootstrap_ci(flags.cpu().numpy(), n=bootstrap)
        out[f'R@{k}'] = (acc, lo, hi)
    return out

# ------------------ annotations ------------------
LANG_KEYS = ['language','lang','lng']
ID_KEYS   = ['image_id','img_id','id','image','img','file_name','filename']
CAP_KEYS  = ['caption','text','sentence','sentences','utterance']

KNOWN_LANG_DIRS = set('ar bn cs da de el en es et fi fr hi hu id it ja ko nl pl pt ru sv tr uk vi zh'.split())

def infer_lang_from_path(p: str):
    # expecting .../annotations/<lang>/<file>
    parent = os.path.basename(os.path.dirname(p))
    return parent if parent in KNOWN_LANG_DIRS else None

def load_annotations_recursive(root: str, split: str):
    paths = []
    if os.path.isdir(root):
        paths = glob.glob(os.path.join(root, '**', '*'), recursive=True)
    else:
        paths = [root]
    files = [p for p in paths if os.path.isfile(p) and os.path.splitext(p)[1].lower() in ('.jsonl','.json','.csv','.tsv')]
    # filter by split (e.g., "test")
    base = [p for p in files if split.lower() in os.path.basename(p).lower()]
    if not base:
        # if split filtering removed everything, fall back to all files (warn later)
        base = files

    recs = []
    for p in base:
        ext = os.path.splitext(p)[1].lower()
        lang_hint = infer_lang_from_path(p)
        try:
            if ext == '.jsonl':
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        obj = json.loads(line)
                        obj['_srcfile'] = p
                        if lang_hint and not any(k in obj for k in LANG_KEYS):
                            obj['language'] = lang_hint
                        recs.append(obj)
            elif ext == '.json':
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        obj['_srcfile'] = p
                        if lang_hint and not any(k in obj for k in LANG_KEYS):
                            obj['language'] = lang_hint
                        recs.append(obj)
                elif isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                    for obj in data['data']:
                        obj['_srcfile'] = p
                        if lang_hint and not any(k in obj for k in LANG_KEYS):
                            obj['language'] = lang_hint
                        recs.append(obj)
            else:
                import csv
                dialect = 'excel' if ext == '.csv' else 'excel-tab'
                with open(p, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, dialect=dialect)
                    for obj in reader:
                        obj['_srcfile'] = p
                        if lang_hint and not any(k in obj for k in LANG_KEYS):
                            obj['language'] = lang_hint
                        recs.append(obj)
        except Exception:
            continue
    return recs, base

def pick(d: dict, keys: List[str]):
    for k in keys:
        if k in d and d[k] not in (None, ''):
            return d[k]
    return None

# -------------------- data prep --------------------
def build_coco_map(coco_root: str):
    m = {}
    if coco_root and os.path.isdir(coco_root):
        for p in glob.glob(os.path.join(coco_root, '*.jpg')):
            mobj = re.search(r'(\d{12})', os.path.basename(p))
            if mobj:
                m[int(mobj.group(1))] = p
    return m

def build_flickr_map(flickr_root: str):
    m = {}
    if flickr_root and os.path.isdir(flickr_root):
        for p in glob.glob(os.path.join(flickr_root, '*.jpg')):
            stem = os.path.splitext(os.path.basename(p))[0]
            m[stem] = p
    return m

def resolve_image_path(rec: dict, coco_map: Dict[int,str], flickr_map: Dict[str,str], coco_root: str, flickr_root: str):
    img_id = pick(rec, ID_KEYS)
    src = pick(rec, ['source','dataset','split','ds']) or ''
    # direct path?
    if isinstance(img_id, str) and os.path.isabs(img_id) and os.path.exists(img_id):
        return img_id
    # file-like
    if isinstance(img_id, str) and img_id.lower().endswith(('.jpg','.jpeg','.png')):
        # try under roots
        for root in (coco_root, flickr_root):
            if root and os.path.exists(os.path.join(root, img_id)):
                return os.path.join(root, img_id)
        if os.path.isabs(img_id) and os.path.exists(img_id):
            return img_id
    s = src.lower()
    # COCO by 12-digit id
    mobj = None
    if isinstance(img_id, (int, np.integer, np.int64)):
        mobj = int(img_id)
    elif isinstance(img_id, str):
        try:
            mobj = int(img_id)
        except Exception:
            m12 = re.search(r'(\d{12})', img_id)
            if m12: mobj = int(m12.group(1))
    if (('coco' in s) or isinstance(mobj, int)) and isinstance(mobj, int) and mobj in coco_map:
        return coco_map[mobj]
    # Flickr by stem
    stem = None
    if isinstance(img_id, str):
        stem = os.path.splitext(os.path.basename(img_id))[0]
    elif isinstance(img_id, int):
        stem = str(img_id)
    if stem and stem in flickr_map:
        return flickr_map[stem]
    # last resort
    raise FileNotFoundError(f'Cannot resolve image path (src={src}, image_id={img_id})')

class ImageListDataset(Dataset):
    def __init__(self, paths: List[str], preprocess):
        self.paths = paths; self.preprocess = preprocess
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return self.preprocess(img), i

# -------------------- main --------------------
def parse_args():
    ap = argparse.ArgumentParser(description='xFlickr&CO retrieval evaluation (recursive annotations)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--clip_model_name', type=str, default='ViT-B/32')
    ap.add_argument('--mclip_model_path', type=str, required=True)
    ap.add_argument('--tomclip_model_path', type=str, required=True)
    ap.add_argument('--coco_root', type=str, default='/xFlickrCO/coco_dataset/val2014')
    ap.add_argument('--flickr_root', type=str, default='/xFlickrCO/flickr30k_images')
    ap.add_argument('--ann_path', type=str, default='/xFlickrCO/annotations', help='Root of annotations (directory or file)')
    ap.add_argument('--split', type=str, default='test', help='Which split files to use (e.g., test). If not found, falls back to all files.')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--bootstrap', type=int, default=0)
    ap.add_argument('--out', type=str, default='xflickrco_results.csv')
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    set_deterministic(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load(args.clip_model_name, device=device)
    clip_model.eval().requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-B-32')

    def embed_texts(texts: List[str], model: SentenceModelWithLinearTransformation):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=77)
        ids = inputs['input_ids'].to(device); attn = inputs.get('attention_mask', None)
        attn = attn.to(device) if attn is not None else None
        with torch.no_grad():
            z = model(ids, attn).float(); z = normalize(z)
        return z.cpu()

    # load (M)CLIP
    mclip = SentenceModelWithLinearTransformation('xlm-roberta-large', target_embedding_dim=512)
    mclip.load_state_dict(torch.load(args.mclip_model_path, map_location=device))
    mclip.to(device).eval().requires_grad_(False)

    tom = SentenceModelWithLinearTransformation('xlm-roberta-large', target_embedding_dim=512)
    tom.load_state_dict(torch.load(args.tomclip_model_path, map_location=device))
    tom.to(device).eval().requires_grad_(False)

    coco_map = build_coco_map(args.coco_root)
    flickr_map = build_flickr_map(args.flickr_root)

    recs, used_files = load_annotations_recursive(args.ann_path, args.split)
    if not recs:
        raise RuntimeError(f'No annotation records found under {args.ann_path}. (Checked recursively; split="{args.split}")')
    # warn if split didn't filter anything
    has_split = any(args.split.lower() in os.path.basename(f).lower() for f in used_files)

    # group by language
    def get_lang(r):
        for k in LANG_KEYS:
            if k in r and r[k]:
                return r[k]
        return None
    def get_cap(r):
        for k in CAP_KEYS:
            if k in r and r[k]:
                return r[k]
        return None

    by_lang = defaultdict(list)
    for r in recs:
        lang = get_lang(r)
        cap  = get_cap(r)
        if not cap: continue
        try:
            imgp = resolve_image_path(r, coco_map, flickr_map, args.coco_root, args.flickr_root)
        except Exception:
            continue
        if not lang:
            # fallback from file path
            lang = infer_lang_from_path(r.get('_srcfile','')) or 'unknown'
        by_lang[lang].append((imgp, cap))

    if not by_lang:
        raise RuntimeError('Parsed annotations but none resolved to (img, caption) pairs. Check roots/ids.')

    # helpers
    def embed_images(paths: List[str]):
        ds = ImageListDataset(paths, preprocess)
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        feats = []
        with torch.no_grad():
            for x,_ in tqdm(dl, desc='Encode images'):
                x = x.to(device)
                f = clip_model.encode_image(x).float()
                feats.append(normalize(f).cpu())
        return torch.cat(feats, dim=0)

    def embed_texts_clip(texts: List[str]):
        toks = clip.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            z = clip_model.encode_text(toks).float()
            z = normalize(z)
        return z.cpu()

    # evaluate per language
    rows = []
    for lang in sorted(by_lang.keys(), key=lambda s: s.lower()):
        pairs = OrderedDict()
        for imgp, cap in by_lang[lang]:
            if imgp not in pairs:
                pairs[imgp] = cap[0]
        img_paths = list(pairs.keys()); texts = list(pairs.values())
        if len(img_paths) == 0:
            continue

        imgs = embed_images(img_paths)
        txt_c = embed_texts_clip(texts)
        txt_m = embed_texts(texts, mclip)
        txt_t = embed_texts(texts, tom)

        S_ir_c = txt_c @ imgs.T
        S_ir_m = txt_m @ imgs.T
        S_ir_t = txt_t @ imgs.T

        S_tr_c = imgs @ txt_c.T
        S_tr_m = imgs @ txt_m.T
        S_tr_t = imgs @ txt_t.T

        for tag, S in [('CLIP-IR', S_ir_c), ('MCLIP-IR', S_ir_m), ('ToMCLIP-IR', S_ir_t),
                       ('CLIP-TR', S_tr_c), ('MCLIP-TR', S_tr_m), ('ToMCLIP-TR', S_tr_t)]:
            res = compute_recalls(S, ks=(1,5,10), bootstrap=args.bootstrap)
            for k in (1,5,10):
                acc, lo, hi = res[f'R@{k}']
                rows.append({'language': lang, 'model': tag, 'k': k, 'R@k': acc, 'CI_lo': lo, 'CI_hi': hi, 'pairs': len(img_paths)})

        print(f"[{lang}] pairs={len(img_paths)}  IR/CLIP R@1={human_pct((S_ir_c.topk(1,dim=1).indices==torch.arange(S_ir_c.size(0)).unsqueeze(1)).any(dim=1).float().mean().item())}")

    # save
    import pandas as pd
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.out, index=False, encoding='utf-8')
        print(f"Saved → {args.out}")
    else:
        print("No rows saved (empty).")

if __name__ == '__main__':
    main()