# FlakeAI - Sıfırdan Eğitilen AI Modeli

**400-700M parametre, CPU-optimized, kendi AI modeliniz**

## Özellikler
- Sıfırdan eğitilmiş Transformer modeli
- Kod yazma, metin üretimi, fotoğraf analizi
- Masaüstü uygulaması (.deb + .exe)
- Web arayüzü
- CPU-optimized (GPU gerekmez)

## Kurulum

### Arch Linux
```bash
git clone https://github.com/KULLANICI/FlakeAI.git
cd FlakeAI
chmod +x install-arch.sh
./install-arch.sh
```

### Debian/Ubuntu
```bash
sudo dpkg -i flakeai_1.0_amd64.deb
```

### Windows
```
FlakeAI-Setup.exe çalıştır
```

### Kaynaktan
```bash
pip install -r requirements.txt
python -m app.desktop.main
```

## Kullanım

### Terminal
```bash
flakeai "Hello world"           # Metin üret
flakeai --image photo.jpg       # Fotoğraf analiz
flakeai --code "sorting"        # Kod yaz
```

### Web
```bash
python -m app.web.server
# Tarayıcıda: http://localhost:8080
```

### Masaüstü
```bash
flakeai-gui
```

## Eğitim
```bash
python -m training.train --config configs/base.yaml
```

## Lisans
MIT License
