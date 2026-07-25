"""
FlakeAI - Ana Giriş Noktası
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="FlakeAI - Sıfırdan eğitilen AI modeli",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py "Hello world"
  python main.py --mode chat
  python main.py --mode web
  python main.py --mode gui
  python main.py --mode train --data data/
        """
    )
    
    parser.add_argument('prompt', nargs='?', help='Giriş metni')
    parser.add_argument('--mode', choices=['text', 'chat', 'web', 'gui', 'train'],
                       default='text', help='Çalışma modu')
    parser.add_argument('--model', type=str, default=None, help='Model yolu')
    parser.add_argument('--data', type=str, default=None, help='Veri seti yolu (eğitim)')
    parser.add_argument('--max-tokens', type=int, default=512, help='Maksimum token')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sıcaklık')
    parser.add_argument('--port', type=int, default=8080, help='Web sunucu portu')
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        from app.desktop.main import main as gui_main
        gui_main()
    
    elif args.mode == 'web':
        from app.web.server import run_server
        run_server(port=args.port)
    
    elif args.mode == 'chat':
        from inference.engine import FlakeInference
        
        engine = FlakeInference(args.model)
        
        print("FlakeAI Sohbet Modu")
        print("Çıkmak için 'quit' yazın\n")
        
        history = []
        while True:
            user_input = input("Sen: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            response = engine.chat(user_input, history)
            print(f"FlakeAI: {response}\n")
            
            history.append({'user': user_input, 'assistant': response})
    
    elif args.mode == 'train':
        if not args.data:
            print("Hata: Eğitim modu için --data gerekli")
            sys.exit(1)
        
        from training.train import main as train_main
        sys.argv = ['train', '--data', args.data]
        train_main()
    
    else:
        if not args.prompt:
            print("Hata: Metin modu için prompt gerekli")
            sys.exit(1)
        
        from inference.engine import FlakeInference
        
        engine = FlakeInference(args.model)
        response = engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(response)


if __name__ == '__main__':
    main()
