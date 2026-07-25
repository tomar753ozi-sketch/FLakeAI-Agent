#!/bin/bash
# FlakeAI - .deb Paket Oluşturucu

set -e

PACKAGE="flakeai"
VERSION="1.0.0"
ARCH="amd64"
BUILD_DIR="dist/deb/build"

GREEN='\033[0;32m'
NC='\033[0m'

log() { echo -e "${GREEN}[DEB]${NC} $1"; }

# Temizle
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/DEBIAN"
mkdir -p "$BUILD_DIR/usr/bin"
mkdir -p "$BUILD_DIR/usr/share/flakeai"

# Dosyaları kopyala
log "Dosyalar kopyalanıyor..."
cp -r model tokenizer training inference app "$BUILD_DIR/usr/share/flakeai/"
cp requirements.txt main.py "$BUILD_DIR/usr/share/flakeai/"
cp -r configs "$BUILD_DIR/usr/share/flakeai/"

# Ana komutları oluştur
cat > "$BUILD_DIR/usr/bin/flakeai" << 'EOF'
#!/bin/bash
cd /usr/share/flakeai
python3 main.py "$@"
EOF
chmod 755 "$BUILD_DIR/usr/bin/flakeai"

cat > "$BUILD_DIR/usr/bin/flakeai-gui" << 'EOF'
#!/bin/bash
cd /usr/share/flakeai
python3 main.py --mode gui "$@"
EOF
chmod 755 "$BUILD_DIR/usr/bin/flakeai-gui"

cat > "$BUILD_DIR/usr/bin/flakeai-web" << 'EOF'
#!/bin/bash
cd /usr/share/flakeai
python3 main.py --mode web "$@"
EOF
chmod 755 "$BUILD_DIR/usr/bin/flakeai-web"

# control dosyası
cat > "$BUILD_DIR/DEBIAN/control" << EOF
Package: $PACKAGE
Version: $VERSION
Section: devel
Priority: optional
Architecture: $ARCH
Depends: python3 (>= 3.10), python3-pip, python3-venv
Maintainer: FlakeAI Team <team@flakeai.dev>
Homepage: https://github.com/tomar753ozi-sketch/FlakeAI-Agent
Installed-Size: $(du -sk "$BUILD_DIR/usr" | cut -f1)
Description: FlakeAI Agent - Sıfırdan eğitilen AI modeli
 FlakeAI, sıfırdan eğitilmiş 400-700M parametrelik
 Transformer modelidir. Kod yazma, metin üretimi
 ve fotoğraf analizi yapabilir.
 .
 - Masaüstü uygulaması (PyQt6)
 - Web arayüzü (FastAPI)
 - CPU-optimized
EOF

# postinst
cat > "$BUILD_DIR/DEBIAN/postinst" << 'POSTINST'
#!/bin/bash
echo "FlakeAI kuruluyor..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || true
pip3 install transformers datasets accelerate peft || true
pip3 install flask fastapi uvicorn || true
pip3 install pyqt6 pillow numpy tqdm pyyaml requests sentencepiece protobuf || true
echo "Kurulum tamamlandı!"
echo "Kullanım: flakeai 'Hello world'"
POSTINST
chmod 755 "$BUILD_DIR/DEBIAN/postinst"

# postrm
cat > "$BUILD_DIR/DEBIAN/postrm" << 'POSTRM'
#!/bin/bash
echo "FlakeAI kaldırıldı."
POSTRM
chmod 755 "$BUILD_DIR/DEBIAN/postrm"

# .deb oluştur
log ".deb paketi oluşturuluyor..."
mkdir -p dist
dpkg-deb --build "$BUILD_DIR" "dist/flakeai_${VERSION}_${ARCH}.deb"

log "Oluşturuldu: dist/flakeai_${VERSION}_${ARCH}.deb"
