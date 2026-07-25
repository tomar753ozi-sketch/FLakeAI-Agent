#!/bin/bash
# FlakeAI - Arch Linux Kurulum Scripti

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

INSTALL_DIR="$HOME/.local/share/flakeai"
BIN_DIR="$HOME/.local/bin"
CONFIG_DIR="$HOME/.config/flakeai"

log() { echo -e "${GREEN}[FlakeAI]${NC} $1"; }
warn() { echo -e "${YELLOW}[UYARI]${NC} $1"; }
err() { echo -e "${RED}[HATA]${NC} $1"; exit 1; }

# Bağımlılıkları kur
install_deps() {
    log "Bağımlılıklar kuruluyor..."
    
    # Python 3.11 kur (eğer yoksa)
    if ! command -v python3.11 &> /dev/null; then
        warn "Python 3.11 kurulu değil, kuruluyor..."
        sudo pacman -S --needed --noconfirm python311 python311-pip
    fi
    
    # Python paketleri
    pip3.11 install --user --upgrade pip
    pip3.11 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip3.11 install --user transformers datasets accelerate peft
    pip3.11 install --user flask fastapi uvicorn
    pip3.11 install --user pyqt6 pillow numpy tqdm pyyaml requests sentencepiece protobuf
}

# FlakeAI'ı kur
install_flakeai() {
    log "FlakeAI kuruluyor..."
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$BIN_DIR"
    mkdir -p "$CONFIG_DIR"
    
    # Dosyaları kopyala
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/"
    
    # Ana komut
    cat > "$BIN_DIR/flakeai" << 'EOF'
#!/bin/bash
python3.11 -m inference.engine "$@"
EOF
    chmod +x "$BIN_DIR/flakeai"
    
    # GUI komutu
    cat > "$BIN_DIR/flakeai-gui" << 'EOF'
#!/bin/bash
python3.11 -m app.desktop.main "$@"
EOF
    chmod +x "$BIN_DIR/flakeai-gui"
    
    # Web komutu
    cat > "$BIN_DIR/flakeai-web" << 'EOF'
#!/bin/bash
python3.11 -m app.web.server "$@"
EOF
    chmod +x "$BIN_DIR/flakeai-web"
    
    log "Kurulum tamamlandı!"
}

# PATH'e ekle
setup_path() {
    log "PATH ayarlanıyor..."
    
    if ! grep -q "$HOME/.local/bin" ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        warn "PATH güncellendi. Yeni terminal açın."
    fi
}

# .desktop dosyası oluştur
create_desktop_entry() {
    log "Desktop entry oluşturuluyor..."
    
    cat > "$HOME/.local/share/applications/flakeai.desktop" << EOF
[Desktop Entry]
Name=FlakeAI
Comment=Sıfırdan eğitilen AI modeli
Exec=$BIN_DIR/flakeai-gui
Icon=flakeai
Terminal=false
Type=Application
Categories=Development;AI;
EOF
}

# Main
main() {
    echo ""
    echo "═══════════════════════════════════════"
    echo "    FlakeAI Kurulum"
    echo "═══════════════════════════════════════"
    echo ""
    
    install_deps
    install_flakeai
    setup_path
    create_desktop_entry
    
    echo ""
    echo "═══════════════════════════════════════"
    echo "    Kurulum Tamamlandı!"
    echo "═══════════════════════════════════════"
    echo ""
    echo "Kullanım:"
    echo "  flakeai 'Hello world'        # Terminal"
    echo "  flakeai-gui                   # Masaüstü"
    echo "  flakeai-web                   # Web arayüzü"
    echo ""
    echo "Örnek:"
    echo "  flakeai 'Write a sorting algorithm'"
    echo "  flakeai --image photo.jpg 'What is this?'"
    echo ""
}

main "$@"
