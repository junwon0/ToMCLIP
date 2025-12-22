import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as arrDataset
from tqdm import tqdm
import os
import transformers
from datasets import load_dataset
from TrainingModel_pt import SentenceModelWithLinearTransformation
from earlystopping import EarlyStopping
from topology_loss import TopologyLossCalculator
import time
from datetime import timedelta
import clip
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
    parser = argparse.ArgumentParser(description="Train SentenceModel with Linear Transformation")
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number (default: 0)')
    parser.add_argument('--pretrained_experiment', type=str, default=None, help='the route of pretrained experiment for additional training')
    parser.add_argument('--datause', type=float, default=1.0, help='low resource condition')
    parser.add_argument('--data_korean', type=str2bool, default=False, help='korean caption dataset use status')
    parser.add_argument('--experiment', type=str, default='mclip', help='experiment name for saving')
    parser.add_argument('--tda_loss', type=str, default=None, help='what kind of loss of tda features')
    parser.add_argument('--swd_k', type=int, default=50, help='number of projections of sliced wasserstein distance')
    parser.add_argument('--std_scale', type=float, default=0.5, help='std scale of graph sparsification')
    parser.add_argument('--dm_loss', type=str, default=None, help='what kind of loss of tda features')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (default: 1e-5)')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--patience', type=int, default=5, help='earlystopping patience')
    parser.add_argument('--mse_co', type=float, default=1, help='Coefficient for mse loss')
    parser.add_argument('--alpha', type=float, default=0.01, help='Coefficient for dm loss')
    parser.add_argument('--beta', type=float, default=0.01, help='Coefficient for 0-dim TDA loss')
    parser.add_argument('--gamma', type=float, default=0.01, help='Coefficient for 1-dim TDA loss')
    parser.add_argument('--pi_weight', type=str, default='constant', help='Weight function for persistence image (constant, linear)')
    parser.add_argument('--pi_bandwidth', type=float, default=0.01, help='Bandwidth for persistence image Gaussian kernel')

    args = parser.parse_args()
    return args

def load_text_translations(data_korean=False):
    dataset = load_dataset('M-CLIP/ImageCaptions-7M-Translations', split='train')
    if data_korean:
        print('korean data is utilized')
        df = dataset.to_pandas()
        df_ko = pd.read_csv("sampled_for_translation_korean_translated.csv", encoding='utf-8')   
        ko_indices = (df_ko['index']-1).tolist()
        ko_captions = df_ko['caption_ko'].tolist()

        for idx, new_caption in zip(ko_indices, ko_captions):
            df.at[idx, 'caption_multi'] = new_caption
            df.at[idx, 'multi_language_code'] = 'ko'
            df.at[idx, 'multi_language_name'] = 'korean'
        dataset = arrDataset.from_pandas(df)
    return dataset

def load_target_embeddings(image_base="Vit-B-32", validation_size=5000, args=None):
    num_data = 2000000
    if args is not None:
        if not args.datause == 1.0:
            num_data = int( 2000000 * args.datause )
    train_samples = load_dataset(
        'M-CLIP/ImageCaptions-7M-Embeddings', 
        image_base, 
        split=f'train[{validation_size}:{num_data}]', 
        trust_remote_code=True  
    )
    val_samples = load_dataset(
        'M-CLIP/ImageCaptions-7M-Embeddings', 
        image_base, 
        split=f'train[:{validation_size}]', 
        trust_remote_code=True 
    )
    embedding_dim = len(train_samples[0]['embedding'][0])
    return train_samples, val_samples, embedding_dim

class ImageTextEmbeddingDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, target_captions):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.target_captions = target_captions

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        text = self.target_captions[self.hf_dataset[idx]['id']]['caption_multi']
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        embedding = torch.tensor(self.hf_dataset[idx]['embedding'], dtype=torch.float)
        return {
            'input_ids': inputs['input_ids'].squeeze(0),       # (seq_len)
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'embedding': embedding
        }
    
class ImageTextEmbeddingDataset_7M(Dataset):
    def __init__(self, target_captions, tokenizer, clip_model_name, device):
        self.target_captions = target_captions
        self.tokenizer = tokenizer
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        self.device = device

    def __len__(self):
        return len(self.target_captions)

    def __getitem__(self, idx):
        caption_multi = self.target_captions[idx]['caption_multi']
        caption_en = self.target_captions[idx]['caption']

        inputs = self.tokenizer(
            caption_multi,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=77
        )

        with torch.no_grad():
            text_tokens = clip.tokenize([caption_en]).to(self.device)
            clip_embedding = self.clip_model.encode_text(text_tokens).squeeze(0).cpu()

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'embedding': clip_embedding
        }
    
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    embeddings = torch.stack([item['embedding'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'embedding': embeddings
    }

def create_optimizer_and_scheduler(model, num_train_steps, num_warmup_steps, lr=1e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = transformers.get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )
    return optimizer, scheduler


def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, log_path, checkpoint_path, tda_loss=None, pretrain_val_loss=None, save_interval=1000, args=None):
    loss_fn = nn.MSELoss()
    model.to(device)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=checkpoint_path)

    if pretrain_val_loss is None:
        best_val_loss = float('inf')
    else:
        best_val_loss = pretrain_val_loss
    best_epoch = -1  

    
    for epoch in range(num_epochs):
        with open(log_path, 'a') as f_log:
            start_time = time.time()
            model.train()
            train_loss = 0.0
            mclip_loss = 0.0
            dm_loss_value = 0.0
            zerodim_tda_loss = 0.0
            onedim_tda_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_embedding = batch['embedding'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                target_embedding = target_embedding.squeeze(1)
                loss = args.mse_co * loss_fn(outputs, target_embedding)
                mclip_loss += loss.item() / args.mse_co

                if args.dm_loss is not None:
                    outputs_dm = model.tda_module.compute_distmat(outputs)
                    target_embedding_dm = model.tda_module.compute_distmat(target_embedding)
                    dm = loss_fn(outputs_dm, target_embedding_dm)
                    loss += args.alpha * dm
                    dm_loss_value += dm.item()

                if tda_loss is not None:
                    outputs_pds = model.tda_module(outputs)
                    target_embedding_pds = model.tda_module(target_embedding)
                    if tda_loss == 'swd':
                        zerodim, onedim = model.tda_module.swd_loss(outputs_pds, target_embedding_pds)
                    elif tda_loss == 'pi':
                        zerodim, onedim = model.tda_module.pi_loss(outputs_pds, target_embedding_pds)
                    loss += args.beta * zerodim
                    loss += args.gamma * onedim

                    zerodim_tda_loss += zerodim.item()
                    onedim_tda_loss += onedim.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))

            mclip_loss = mclip_loss / len(train_loader)
            dm_loss_value = dm_loss_value / len(train_loader)
            zerodim_tda_loss = zerodim_tda_loss / len(train_loader)
            onedim_tda_loss = onedim_tda_loss / len(train_loader)
            val_loss, mclip_val_loss, mclip_dm_loss, zerodim_val_loss, onedim_val_loss = evaluate(model, val_loader, device, args.dm_loss, tda_loss)

            if val_loss < best_val_loss-1e-8:
                best_val_loss = val_loss
                best_epoch = epoch

            log_msg = f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, mclip Loss: {mclip_loss:.4f}, dm_loss: {dm_loss_value:.4f}, zerodim Loss: {zerodim_tda_loss:.4f}, onedim Loss: {onedim_tda_loss:.4f}, Elapsed Time: {elapsed_str}"
            print(log_msg)
            f_log.write(log_msg + '\n')
            log_msg = f"Validation Loss: {val_loss:.4f}, mclip Loss: {mclip_val_loss:.4f}, dm_loss: {mclip_dm_loss:.4f}, zerodim Loss: {zerodim_val_loss:.4f}, onedim Loss: {onedim_val_loss:.4f}"
            print(log_msg)
            f_log.write(log_msg + '\n')

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                f_log.write(f"Early stopping at epoch {epoch}\n")
                break

            if (epoch + 1) % save_interval == 0:
                save_path = checkpoint_path.replace('best_model.pt', f'checkpoint_epoch_{epoch+1}.pt')
                torch.save(model.state_dict(), save_path)
                print(f"Checkpoint saved at {save_path}")
                f_log.write(f"Checkpoint saved at {save_path}\n")

    with open(log_path, 'a') as f_log:
        f_log.write(f"\nBest Epoch: {best_epoch}\n")
        f_log.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        f_log.write(f"Experiment ended at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")        
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Validation Loss: {best_val_loss:.4f}")


def evaluate(model, val_loader, device, dm_loss=None, tda_loss=None):
    model.eval()
    loss_fn = nn.MSELoss()
    mclip_loss = 0.0
    dm_loss_value = 0.0
    zerodim_tda_loss = 0.0
    onedim_tda_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_embedding = batch['embedding'].to(device)

            outputs = model(input_ids, attention_mask)
            target_embedding = target_embedding.squeeze(1)
            loss = loss_fn(outputs, target_embedding)
            mclip_loss += loss.item()

            if dm_loss is not None:
                outputs_dm = model.tda_module.compute_distmat(outputs)
                target_embedding_dm = model.tda_module.compute_distmat(target_embedding)
                dm = loss_fn(outputs_dm, target_embedding_dm)
                dm_loss_value += dm.item()

            if tda_loss is not None:
                outputs_pds = model.tda_module(outputs)
                target_embedding_pds = model.tda_module(target_embedding)
                if tda_loss == 'swd':
                    zerodim, onedim = model.tda_module.swd_loss(outputs_pds, target_embedding_pds)
                elif tda_loss == 'pi':
                    zerodim, onedim = model.tda_module.pi_loss(outputs_pds, target_embedding_pds)

                zerodim_tda_loss += zerodim.item()
                onedim_tda_loss += onedim.item()

    total_loss = args.mse_co * mclip_loss + args.alpha * dm_loss_value + args.beta * zerodim_tda_loss + args.gamma * onedim_tda_loss

    avg_loss = total_loss / len(val_loader)

    avg_mclip_loss = mclip_loss / len(val_loader)
    avg_dm_loss = dm_loss_value / len(val_loader)
    avg_zerodim_tda_loss = zerodim_tda_loss / len(val_loader)
    avg_onedim_tda_loss = onedim_tda_loss / len(val_loader)
    return avg_loss, avg_mclip_loss, avg_dm_loss, avg_zerodim_tda_loss, avg_onedim_tda_loss
    

def run_training(gpu_id, log_path, checkpoint_path, tda_loss=None, args=None):
    num_validation_samples = 5000
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    num_train_steps = 99999999
    num_warmup_steps = 1000

    model_base = 'xlm-roberta-large'
    image_base = "Vit-B-32"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_base)
    target_captions = load_text_translations(args.data_korean)
    train_embeddings, val_embeddings, embedding_dim = load_target_embeddings(image_base=image_base, validation_size=num_validation_samples,args=args)

    if not args.datause == 1.0:
        with open(log_path, 'a') as f_log:
            f_log.write(f'data len: train {len(train_embeddings)}, val {len(val_embeddings)}\n')

    train_dataset = ImageTextEmbeddingDataset(train_embeddings, tokenizer, target_captions)
    val_dataset = ImageTextEmbeddingDataset(val_embeddings, tokenizer, target_captions)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SentenceModelWithLinearTransformation(model_base, embedding_dim)
    
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Experiemt log path: {log_path}")
    if tda_loss is not None or args.dm_loss is not None:
        model.tda_module = TopologyLossCalculator(output_device=device,loss=tda_loss, args=args)

    if args.pretrained_experiment is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        with open(log_path, 'a') as f_log:
            f_log.write(f"Addtional experiment started.\n")
            f_log.write(f"Device: {device}\n")
            f_log.write(f"Using korean dataset: {args.data_korean}\n")
            if args.dm_loss:
                f_log.write(f"Using dm loss: {args.dm_loss}\n")
            if tda_loss:
                f_log.write(f"Using TDA loss: {tda_loss}\n")
            f_log.write("\n")
            f_log.write(f"Experiment started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_log.write("\n")

            f_log.write("Arguments:\n")
            for arg, value in vars(args).items():
                f_log.write(f"{arg}: {value}\n")
            f_log.write("\n")

            model.to(device)
            val_loss, mclip_val_loss, dm_val_loss, zerodim_val_loss, onedim_val_loss = evaluate(model, val_loader, device, dm_loss=args.dm_loss, tda_loss=tda_loss)
            log_msg = f"Pretrained model validation Loss: {val_loss:.4f}, mclip Loss: {mclip_val_loss:.4f}, dm Loss: {dm_val_loss:.4f}, zerodim Loss: {zerodim_val_loss:.4f}, onedim Loss: {onedim_val_loss:.4f}"
            print(log_msg)
            f_log.write(log_msg + '\n')
    else:
        with open(log_path, 'w') as f_log:
            f_log.write(f"Experiment started.\n")
            f_log.write(f"Device: {device}\n")
            if args.dm_loss:
                f_log.write(f"Using dm loss: {args.dm_loss}\n")
            if tda_loss:
                f_log.write(f"Using TDA loss: {tda_loss}\n")
            f_log.write("\n")
            f_log.write(f"Experiment started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_log.write("\n")

            f_log.write("Arguments:\n")
            for arg, value in vars(args).items():
                f_log.write(f"{arg}: {value}\n")
            f_log.write("\n")
        val_loss = None

    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_steps, num_warmup_steps, lr=lr)

    train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, log_path, checkpoint_path, tda_loss=tda_loss, pretrain_val_loss=val_loss, args=args)


if __name__ == '__main__':
    args = parse_args()

    if args.pretrained_experiment is None:
        exp_dir = f"experiments/{args.experiment}_data{args.datause}_ko{args.data_korean}_{time.strftime('%Y%m%d')}_{args.lr}"
        if args.dm_loss is not None:
            exp_dir += f'_dm_loss_{args.alpha}'
        if args.tda_loss is not None:
            exp_dir += f'_{args.tda_loss.lower()}_{args.mse_co}_{args.beta}_{args.gamma}'

    else:
        exp_dir = args.pretrained_experiment
    
    args.log_path = os.path.join(exp_dir, 'train.log')
    args.checkpoint_path = os.path.join(exp_dir, 'best_model.pt')
    
    os.makedirs(exp_dir, exist_ok=True)

    run_training(args.gpu, args.log_path, args.checkpoint_path, args.tda_loss, args=args)