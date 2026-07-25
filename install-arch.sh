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
    
    sudo pacman -S --needed --noconfirm python python-pip git base-devel
    
    log "Virtual environment oluşturuluyor..."
    python -m venv "$VENV_DIR"
    
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
    
    # GUI launcher
    cat > "$BIN_DIR/flakeai-gui" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
cd "$INSTALL_DIR"
python main.py --mode gui "\$@"
EOF
    chmod +x "$BIN_DIR/flakeai-gui"
    
    # Web launcher
    cat > "$BIN_DIR/flakeai-web" << EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
cd "$INSTALL_DIR"
python main.py --mode web "\$@"
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
    log "Uygulama ikonu oluşturuluyor..."
    
    mkdir -p "$HOME/.local/share/applications"
    mkdir -p "$HOME/.local/share/icons/hicolor/256x256/apps"
    
    # Desktop entry
    cat > "$HOME/.local/share/applications/flakeai.desktop" << EOF
[Desktop Entry]
Name=FlakeAI
GenericName=AI Assistant
Comment=Sıfırdan eğitilen AI modeli
Exec=$BIN_DIR/flakeai-gui
Icon=python
Terminal=false
Type=Application
Categories=Development;Utility;
Keywords=ai;assistant;
StartupWMClass=flakeai
EOF
    
    chmod +x "$HOME/.local/share/applications/flakeai.desktop"
    
    # Desktop database güncelle
    update-desktop-database ~/.local/share/applications/ 2>/dev/null || true
    
    log "Uygulama menüye eklendi!"
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
    echo "Uygulamayı açmak için:"
    echo "  - Uygulama menüsünden 'FlakeAI' ara"
    echo "  - Veya terminalden: flakeai-gui"
    echo ""
}

main "$@"
