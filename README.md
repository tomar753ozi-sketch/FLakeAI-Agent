# FlakeAI Agent

**Sıfırdan eğitilen AI modeli - 400-700M parametre**

## İndir

| Platform | Link | Boyut |
|----------|------|-------|
| **Windows (.exe)** | [FlakeAI-Setup.exe](https://github.com/tomar753ozi-sketch/FlakeAI-Agent/releases/download/v1.0/FlakeAI-Setup.exe) | ~50MB |
| **Debian/Ubuntu (.deb)** | [flakeai_1.0_amd64.deb](https://github.com/tomar753ozi-sketch/FlakeAI-Agent/releases/download/v1.0/flakeai_1.0_amd64.deb) | ~50MB |
| **Arch Linux** | `git clone` + kurulum | - |
| **Kaynak Kod** | [FlakeAI-Agent.zip](https://github.com/tomar753ozi-sketch/FlakeAI-Agent/archive/refs/heads/main.zip) | ~20KB |

## Özellikler
- Sıfırdan eğitilmiş Transformer modeli
- Kod yazma, metin üretimi
- Masaüstü uygulaması (PyQt6)
- Web arayüzü (FastAPI)
- CPU-optimized (GPU gerekmez)

## Hızlı Kurulum

### Windows
1. [FlakeAI-Setup.exe](https://github.com/tomar753ozi-sketch/FlakeAI-Agent/releases/download/v1.0/FlakeAI-Setup.exe) indir
2. Çalıştır
3. Kurulumu tamamla

### Debian/Ubuntu
```bash
sudo dpkg -i flakeai_1.0_amd64.deb
sudo apt-get install -f
```

### Arch Linux
```bash
git clone https://github.com/tomar753ozi-sketch/FlakeAI-Agent.git
cd FlakeAI-Agent
chmod +x install-arch.sh
./install-arch.sh
```

### Kaynaktan (Tüm Platformlar)
```bash
git clone https://github.com/tomar753ozi-sketch/FlakeAI-Agent.git
cd FlakeAI-Agent
pip install -r requirements.txt
python main.py --mode web
```

## Kullanım

### Terminal
```bash
python main.py "Hello world"
python main.py --mode chat
```

### Web Arayüzü
```bash
python main.py --mode web
# Tarayıcıda: http://localhost:8080
```

### Masaüstü Uygulaması
```bash
python main.py --mode gui
```

### Eğitim
```bash
python main.py --mode train --data data/
```

## Gereksinimler
- Python 3.10+
- 8GB+ RAM
- 1GB+ disk alanı

## Lisans
MIT License
