#!/bin/bash
# FlakeAI - .exe Oluşturucu (PyInstaller ile)

set -e

GREEN='\033[0;32m'
NC='\033[0m'

log() { echo -e "${GREEN}[EXE]${NC} $1"; }

# PyInstaller kur
log "PyInstaller kuruluyor..."
pip install pyinstaller --quiet

# Temizle
rm -rf build dist/flakeai-windows

# Tek executable oluştur
log ".exe oluşturuluyor..."
pyinstaller \
    --name FlakeAI-Agent \
    --onefile \
    --windowed \
    --icon=NONE \
    --add-data "model:model" \
    --add-data "tokenizer:tokenizer" \
    --add-data "configs:configs" \
    --add-data "requirements.txt:." \
    --hidden-import torch \
    --hidden-import transformers \
    --hidden-import flask \
    --hidden-import fastapi \
    --hidden-import pyqt6 \
    main.py

# Kurulum oluşturucu
log "Kurulum dosyası oluşturuluyor..."
mkdir -p dist/flakeai-windows
cp dist/FlakeAI-Agent.exe dist/flakeai-windows/
cp dist/exe/install.bat dist/flakeai-windows/

# ZIP oluştur
log "ZIP dosyası oluşturuluyor..."
cd dist
zip -r FlakeAI-Agent-Windows.zip flakeai-windows/
cd ..

log "Oluşturuldu:"
log "  dist/FlakeAI-Agent.exe (tek dosya)"
log "  dist/FlakeAI-Agent-Windows.zip (kurulum paketi)"
