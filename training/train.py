"""
FlakeAI - Training Script
CPU-optimized training pipeline
"""

import os
import sys
import json
import time
import math
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import create_model, count_parameters
from tokenizer.tokenizer import FlakeTokenizer


class TextDataset(Dataset):
    """Metin veri seti"""
    
    def __init__(self, texts: List[str], tokenizer: FlakeTokenizer, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
            targets = torch.cat([targets, torch.full((padding_length,), -100, dtype=torch.long)])
        
        return {
            'input_ids': input_ids,
            'targets': targets
        }


class TrainingConfig:
    """Eğitim konfigürasyonu"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.defaults = {
            'model': {
                'variant': 'base',
                'vocab_size': 50000
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 3e-4,
                'weight_decay': 0.1,
                'max_epochs': 100,
                'warmup_steps': 1000,
                'max_grad_norm': 1.0,
                'gradient_accumulation_steps': 4,
                'save_every': 1000,
                'eval_every': 500,
                'output_dir': 'checkpoints'
            },
            'data': {
                'max_length': 2048,
                'num_workers': 0
            },
            'hardware': {
                'device': 'cpu',
                'mixed_precision': False,
                'compile': False
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self._update_dict(self.defaults, config)
    
    def _update_dict(self, base: dict, update: dict):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                self._update_dict(base[key], value)
            else:
                base[key] = value
    
    def __getattr__(self, name: str):
        if name.startswith('_') or name == 'defaults':
            return super().__getattribute__(name)
        
        for group in self.defaults.values():
            if name in group:
                return group[name]
        
        raise AttributeError(f"Config '{name}' bulunamadı")
    
    def get(self, group: str, key: str):
        return self.defaults.get(group, {}).get(key)


class Trainer:
    """Eğitim sınıfı"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.get('hardware', 'device'))
        
        self.model = create_model(config.get('model', 'variant'))
        self.model = self.model.to(self.device)
        
        self.tokenizer = FlakeTokenizer(config.get('model', 'vocab_size'))
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('training', 'learning_rate'),
            weight_decay=config.get('training', 'weight_decay')
        )
        
        self.scaler = None
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        os.makedirs(config.get('training', 'output_dir'), exist_ok=True)
    
    def train(self, dataset: TextDataset):
        """Eğitim döngüsü"""
        batch_size = self.config.get('training', 'batch_size')
        num_workers = self.config.get('data', 'num_workers')
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )
        
        total_steps = len(dataloader) * self.config.get('training', 'max_epochs')
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        grad_accum = self.config.get('training', 'gradient_accumulation_steps')
        max_grad_norm = self.config.get('training', 'max_grad_norm')
        
        print(f"\n{'='*60}")
        print(f"FlakeAI Eğitim Başlatılıyor")
        print(f"{'='*60}")
        print(f"Model: {count_parameters(self.model)/1e6:.1f}M parametre")
        print(f"Cihaz: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation: {grad_accum}")
        print(f"Effective batch size: {batch_size * grad_accum}")
        print(f"{'='*60}\n")
        
        self.model.train()
        
        for epoch in range(self.config.get('training', 'max_epochs')):
            self.epoch = epoch
            epoch_loss = 0
            epoch_steps = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                logits, loss = self.model(input_ids, targets)
                loss = loss / grad_accum
                loss.backward()
                
                epoch_loss += loss.item() * grad_accum
                epoch_steps += 1
                
                if (batch_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm
                    )
                    
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.step += 1
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item() * grad_accum:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                        'step': self.step
                    })
                    
                    if self.step % self.config.get('training', 'save_every') == 0:
                        self.save_checkpoint(f"step_{self.step}")
            
            avg_loss = epoch_loss / epoch_steps
            print(f"\nEpoch {epoch+1} tamamlandı. Ortalama loss: {avg_loss:.4f}")
            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint("best_model")
    
    def save_checkpoint(self, name: str):
        """Checkpoint kaydet"""
        output_dir = self.config.get('training', 'output_dir')
        checkpoint_dir = os.path.join(output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config.defaults
        }, os.path.join(checkpoint_dir, 'model.pt'))
        
        self.tokenizer.save(os.path.join(checkpoint_dir, 'tokenizer.json'))
        
        print(f"Checkpoint kaydedildi: {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Checkpoint yükle"""
        checkpoint = torch.load(
            os.path.join(checkpoint_path, 'model.pt'),
            map_location=self.device
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        tokenizer_path = os.path.join(checkpoint_path, 'tokenizer.json')
        if os.path.exists(tokenizer_path):
            self.tokenizer = FlakeTokenizer.load(tokenizer_path)
        
        print(f"Checkpoint yüklendi: {checkpoint_path}")


def load_text_data(data_path: str) -> List[str]:
    """Metin verisi yükle"""
    texts = []
    
    if os.path.isdir(data_path):
        for file_path in Path(data_path).rglob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.extend([line.strip() for line in f if line.strip()])
    elif os.path.isfile(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    print(f"Yüklenen metin: {len(texts)} örnek")
    return texts


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FlakeAI Training')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                       help='Konfigürasyon dosyası yolu')
    parser.add_argument('--data', type=str, required=True,
                       help='Veri seti yolu')
    parser.add_argument('--resume', type=str, default=None,
                       help='Devam ettirilecek checkpoint')
    
    args = parser.parse_args()
    
    config = TrainingConfig(args.config)
    trainer = Trainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    texts = load_text_data(args.data)
    dataset = TextDataset(texts, trainer.tokenizer, config.get('data', 'max_length'))
    
    trainer.train(dataset)


if __name__ == '__main__':
    main()
