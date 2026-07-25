#!/bin/bash
# FlakeAI - .deb Paket Oluşturucu

set -e

PACKAGE="flakeai"
VERSION="1.0.0"
ARCH="amd64"
BUILD_DIR="dist/deb/build"
INSTALL_DIR="/usr"

GREEN='\033[0;32m'
NC='\033[0m'

log() { echo -e "${GREEN}[DEB]${NC} $1"; }

# Temizle
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/$INSTALL_DIR/flakeai"
mkdir -p "$BUILD_DIR/DEBIAN"

# Dosyaları kopyala
log "Dosyalar kopyalanıyor..."
cp -r model tokenizer training inference app "$BUILD_DIR/$INSTALL_DIR/flakeai/"
cp requirements.txt "$BUILD_DIR/$INSTALL_DIR/flakeai/"

# control dosyasını oluştur
cat > "$BUILD_DIR/DEBIAN/control" << EOF
Package: $PACKAGE
Version: $VERSION
Section: devel
Priority: optional
Architecture: $ARCH
Depends: python3 (>= 3.11), python3-pip
Maintainer: FlakeAI Team <team@flakeai.dev>
Homepage: https://github.com/flakeai/flakeai
Installed-Size: $(du -sk "$BUILD_DIR/$INSTALL_DIR/flakeai" | cut -f1)
Description: FlakeAI - Sıfırdan eğitilen AI modeli
 FlakeAI, sıfırdan eğitilmiş 400-700M parametrelik
 Transformer modelidir. Kod yazma, metin üretimi
 ve fotoğraf analizi yapabilir.
EOF

# postinst script
cat > "$BUILD_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/bash
echo "FlakeAI kuruluyor..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || true
pip3 install transformers datasets accelerate peft || true
pip3 install flask fastapi uvicorn || true
pip3 install pyqt6 pillow numpy tqdm pyyaml requests sentencepiece protobuf || true
echo "Kurulum tamamlandı!"
EOF
chmod 755 "$BUILD_DIR/DEBIAN/postinst"

# .deb oluştur
log ".deb paketi oluşturuluyor..."
dpkg-deb --build "$BUILD_DIR" "dist/flakeai_${VERSION}_${ARCH}.deb"

log "Oluşturuldu: dist/flakeai_${VERSION}_${ARCH}.deb"
