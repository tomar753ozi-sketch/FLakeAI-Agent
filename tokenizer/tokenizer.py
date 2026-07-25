"""
FlakeAI - Tokenizer
BPE tabanlı tokenizer
"""

import os
import json
import re
from typing import List, Dict, Optional
from collections import Counter


class FlakeTokenizer:
    """Basit BPE Tokenizer"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        
        self.special_tokens = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3,
            '<sep>': 4,
            '<cls>': 5,
            '<mask>': 6,
        }
        
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
        
        self.merges: List[tuple] = []
    
    def train(self, texts: List[str], vocab_size: Optional[int] = None):
        """Tokenizer'ı eğit"""
        if vocab_size:
            self.vocab_size = vocab_size
        
        # Kelime frekanslarını hesapla
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            word_freqs.update(words)
        
        # Başlangıç vocabulary'si
        chars = set()
        for word in word_freqs.keys():
            chars.update(word)
        
        for i, char in enumerate(sorted(chars)):
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
        
        # BPE merge'leri
        for _ in range(self.vocab_size - len(self.vocab)):
            if len(self.merges) >= 1000:
                break
            
            pairs = Counter()
            for word, freq in word_freqs.items():
                symbols = list(word)
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += freq
            
            if not pairs:
                break
            
            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)
            
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_token] = idx
                self.inverse_vocab[idx] = new_token
            
            # Merge uygula
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = word
                i = 0
                while i < len(new_word) - 1:
                    if (new_word[i], new_word[i+1]) == best_pair:
                        new_word = new_word[:i] + new_token + new_word[i+2:]
                    else:
                        i += 1
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
        
        print(f"Tokenizer eğitildi: {len(self.vocab)} token")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Metni token'lara çevir"""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.special_tokens['<bos>'])
        
        # Basit tokenization
        words = text.split()
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Character-level fallback
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.special_tokens['<unk>'])
        
        if add_special_tokens:
            tokens.append(self.special_tokens['<eos>'])
        
        return tokens
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Token'ları metne çevir"""
        tokens = []
        for idx in ids:
            token = self.inverse_vocab.get(idx, '<unk>')
            if skip_special_tokens and token in self.special_tokens.values():
                continue
            tokens.append(token)
        
        return ' '.join(tokens)
    
    def save(self, path: str):
        """Tokenizer'ı kaydet"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': [list(m) for m in self.merges],
            'special_tokens': self.special_tokens
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Tokenizer kaydedildi: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FlakeTokenizer':
        """Tokenizer'ı yükle"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(data['vocab_size'])
        tokenizer.vocab = data['vocab']
        tokenizer.inverse_vocab = {int(k): v for k, v in tokenizer.vocab.items()}
        tokenizer.merges = [tuple(m) for m in data['merges']]
        tokenizer.special_tokens = data['special_tokens']
        
        return tokenizer
    
    def __len__(self) -> int:
        return len(self.vocab)
    
    def __contains__(self, token: str) -> bool:
        return token in self.vocab


def train_tokenizer(texts: List[str], vocab_size: int = 50000, save_path: str = 'tokenizer.json'):
    """Tokenizer eğit ve kaydet"""
    tokenizer = FlakeTokenizer(vocab_size)
    tokenizer.train(texts)
    tokenizer.save(save_path)
    return tokenizer
