#!/bin/bash
# FlakeAI - Arch Linux Kurulum Scripti

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

INSTALL_DIR="$HOME/.local/share/flakeai"
BIN_DIR="$HOME/.local/bin"
VENV_DIR="$INSTALL_DIR/venv"

log() { echo -e "${GREEN}[FlakeAI]${NC} $1"; }
warn() { echo -e "${YELLOW}[UYARI]${NC} $1"; }
err() { echo -e "${RED}[HATA]${NC} $1"; exit 1; }

install_deps() {
    log "Bağımlılıklar kuruluyor..."
    
    # Sistem paketleri
    sudo pacman -S --needed --noconfirm python python-pip git base-devel
    
    # Virtual environment oluştur
    log "Virtual environment oluşturuluyor..."
    python -m venv "$VENV_DIR"
    
    # Paketleri venv'e kur
    log "Python paketleri kuruluyor..."
    source "$VENV_DIR/bin/activate"
    
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers datasets accelerate peft
    pip install flask fastapi uvicorn
    pip install pyqt6 pillow numpy tqdm pyyaml requests sentencepiece protobuf
    
    deactivate
}

install_flakeai() {
    log "FlakeAI kuruluyor..."
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$BIN_DIR"
    
    # Dosyaları kopyala
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/"
    
    # Ana komut
    cat > "$BIN_DIR/flakeai" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
cd "$INSTALL_DIR"
python main.py "\$@"
deactivate
EOF
    chmod +x "$BIN_DIR/flakeai"
    
    # GUI komutu
    cat > "$BIN_DIR/flakeai-gui" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
cd "$INSTALL_DIR"
python main.py --mode gui "\$@"
deactivate
EOF
    chmod +x "$BIN_DIR/flakeai-gui"
    
    # Web komutu
    cat > "$BIN_DIR/flakeai-web" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
cd "$INSTALL_DIR"
python main.py --mode web "\$@"
deactivate
EOF
    chmod +x "$BIN_DIR/flakeai-web"
    
    log "Kurulum tamamlandı!"
}

setup_path() {
    log "PATH ayarlanıyor..."
    
    if ! grep -q "$HOME/.local/bin" ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        warn "PATH güncellendi. Yeni terminal açın."
    fi
}

create_desktop_entry() {
    log "Desktop entry oluşturuluyor..."
    
    mkdir -p "$HOME/.local/share/applications"
    
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
}

main "$@"
