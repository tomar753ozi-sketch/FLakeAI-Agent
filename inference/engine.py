"""
FlakeAI - Inference Motoru
Model ile tahmin ve metin üretimi
"""

import torch
import sys
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import FlakeModel, create_model
from tokenizer.tokenizer import FlakeTokenizer


@dataclass
class InferenceConfig:
    """Inference konfigürasyonu"""
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    device: str = 'cpu'


class FlakeInference:
    """FlakeAI inference motoru"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[InferenceConfig] = None
    ):
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        
        if model_path:
            self.load_model(model_path)
        else:
            self.model = create_model("base")
            self.tokenizer = FlakeTokenizer(50000)
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, model_path: str):
        """Model yükle"""
        checkpoint = torch.load(
            f"{model_path}/model.pt",
            map_location=self.device
        )
        
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})
        
        self.model = create_model(model_config.get('variant', 'base'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.tokenizer = FlakeTokenizer.load(f"{model_path}/tokenizer.json")
        
        print(f"Model yüklendi: {model_path}")
    
    @torch.no_grad()
    def generate(self, prompt: str, **kwargs) -> str:
        """Metin üret"""
        config = InferenceConfig(**kwargs) if kwargs else self.config
        
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p
        )
        
        output_text = self.tokenizer.decode(output_ids[0].tolist())
        
        return output_text[len(prompt):]
    
    @torch.no_grad()
    def complete(self, text: str, max_tokens: int = 256) -> str:
        """Metin tamamla"""
        return self.generate(text, max_new_tokens=max_tokens)
    
    @torch.no_grad()
    def chat(self, message: str, history: Optional[List[Dict]] = None) -> str:
        """Sohbet modu"""
        if history is None:
            history = []
        
        prompt = ""
        for h in history[-5:]:
            prompt += f"User: {h['user']}\nAssistant: {h['assistant']}\n"
        prompt += f"User: {message}\nAssistant:"
        
        response = self.generate(prompt, max_new_tokens=256)
        
        return response.strip()
    
    def analyze_image(self, image_path: str) -> str:
        """Fotoğraf analiz (gelecek özellik)"""
        return "Fotoğraf analiz özelliği henüz eklenmedi."
    
    def write_code(self, description: str) -> str:
        """Kod yazma"""
        prompt = f"Write code for: {description}\n\n```python\n"
        code = self.generate(prompt, max_new_tokens=1024)
        
        if "```" not in code:
            code += "\n```"
        
        return code


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FlakeAI Inference')
    parser.add_argument('--model', type=str, default=None,
                       help='Model yolu')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Giriş metni')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maksimum token sayısı')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling sıcaklığı')
    parser.add_argument('--interactive', action='store_true',
                       help='Etkileşimli mod')
    
    args = parser.parse_args()
    
    config = InferenceConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    engine = FlakeInference(args.model, config)
    
    if args.interactive:
        print("FlakeAI Etkileşimli Mod")
        print("Çıkmak için 'quit' yazın\n")
        
        history = []
        while True:
            user_input = input("Sen: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            response = engine.chat(user_input, history)
            print(f"FlakeAI: {response}\n")
            
            history.append({'user': user_input, 'assistant': response})
    else:
        response = engine.generate(args.prompt)
        print(response)


if __name__ == '__main__':
    main()
