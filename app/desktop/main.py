"""
FlakeAI - Masaüstü Uygulaması
PyQt6 tabanlı GUI
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter,
    QMenuBar, QMenu, QStatusBar, QFileDialog, QMessageBox,
    QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QGridLayout, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QAction

from inference.engine import FlakeInference, InferenceConfig


class GenerateThread(QThread):
    """Metin üretimi için arka plan thread'i"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, engine: FlakeInference, prompt: str, config: dict):
        super().__init__()
        self.engine = engine
        self.prompt = prompt
        self.config = config
    
    def run(self):
        try:
            result = self.engine.generate(self.prompt, **self.config)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class FlakeAIWindow(QMainWindow):
    """Ana pencere"""
    
    def __init__(self):
        super().__init__()
        self.engine = None
        self.generate_thread = None
        self.history = []
        
        self.init_ui()
        self.init_engine()
    
    def init_ui(self):
        self.setWindowTitle("FlakeAI v1.0")
        self.setMinimumSize(900, 600)
        
        self.setup_menubar()
        self.setup_ui()
        self.setup_statusbar()
        self.apply_style()
    
    def setup_menubar(self):
        menubar = self.menuBar()
        
        # Dosya menüsü
        file_menu = menubar.addMenu('Dosya')
        
        open_action = QAction('Model Yükle', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.load_model)
        file_menu.addAction(open_action)
        
        save_action = QAction('Sohbeti Kaydet', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_chat)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction('Çıkış', self)
        quit_action.setShortcut('Ctrl+Q')
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Düzen menüsü
        edit_menu = menubar.addMenu('Düzen')
        
        clear_action = QAction('Sohbeti Temizle', self)
        clear_action.setShortcut('Ctrl+L')
        clear_action.triggered.connect(self.clear_chat)
        edit_menu.addAction(clear_action)
        
        # Araçlar menüsü
        tools_menu = menubar.addMenu('Araçlar')
        
        code_action = QAction('Kod Yaz', self)
        code_action.setShortcut('Ctrl+Shift+C')
        code_action.triggered.connect(self.switch_to_code_mode)
        tools_menu.addAction(code_action)
        
        chat_action = QAction('Sohbet Modu', self)
        chat_action.setShortcut('Ctrl+Shift+M')
        chat_action.triggered.connect(self.switch_to_chat_mode)
        tools_menu.addAction(chat_action)
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Üst panel - Ayarlar
        settings_group = QGroupBox("Ayarlar")
        settings_layout = QGridLayout()
        
        # Sıcaklık
        settings_layout.addWidget(QLabel("Sıcaklık:"), 0, 0)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 2.0)
        self.temp_spin.setValue(0.8)
        self.temp_spin.setSingleStep(0.1)
        settings_layout.addWidget(self.temp_spin, 0, 1)
        
        # Max token
        settings_layout.addWidget(QLabel("Max Token:"), 0, 2)
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 4096)
        self.max_tokens_spin.setValue(512)
        self.max_tokens_spin.setSingleStep(64)
        settings_layout.addWidget(self.max_tokens_spin, 0, 3)
        
        # Top-k
        settings_layout.addWidget(QLabel("Top-k:"), 0, 4)
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 100)
        self.top_k_spin.setValue(50)
        settings_layout.addWidget(self.top_k_spin, 0, 5)
        
        # Mod
        settings_layout.addWidget(QLabel("Mod:"), 1, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Sohbet", "Kod Yaz", "Metin Tamamla"])
        settings_layout.addWidget(self.mode_combo, 1, 1)
        
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # Ana içerik
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Sohbet alanı
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 11))
        splitter.addWidget(self.chat_display)
        
        # Giriş alanı
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Mesajınızı yazın...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton("Gönder")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        self.stop_button = QPushButton("Durdur")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_generation)
        input_layout.addWidget(self.stop_button)
        
        splitter.addWidget(input_widget)
        
        main_layout.addWidget(splitter)
    
    def setup_statusbar(self):
        self.statusBar().showMessage("Hazır")
        
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
    
    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #cdd6f4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTextEdit {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 10px;
                font-family: Consolas, monospace;
            }
            QLineEdit {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #89b4fa;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #74c7ec;
            }
            QPushButton:pressed {
                background-color: #89dceb;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
            QLabel {
                color: #cdd6f4;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 3px;
                padding: 4px;
            }
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 3px;
            }
        """)
    
    def init_engine(self):
        """Inference motorunu başlat"""
        try:
            self.engine = FlakeInference()
            self.statusBar().showMessage("Model yüklendi (varsayılan)")
        except Exception as e:
            self.statusBar().showMessage(f"Model yüklenemedi: {e}")
    
    def load_model(self):
        """Model yükle"""
        model_path = QFileDialog.getExistingDirectory(
            self, "Model Dizinini Seçin"
        )
        
        if model_path:
            try:
                self.engine = FlakeInference(model_path)
                self.statusBar().showMessage(f"Model yüklendi: {model_path}")
                self.add_system_message("Model başarıyla yüklendi!")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Model yüklenemedi:\n{e}")
    
    def send_message(self):
        """Mesaj gönder"""
        message = self.input_field.text().strip()
        if not message:
            return
        
        if not self.engine:
            QMessageBox.warning(self, "Uyarı", "Önce bir model yükleyin!")
            return
        
        self.add_user_message(message)
        self.input_field.clear()
        
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.show()
        self.progress.setRange(0, 0)
        
        config = {
            'max_new_tokens': self.max_tokens_spin.value(),
            'temperature': self.temp_spin.value(),
            'top_k': self.top_k_spin.value()
        }
        
        self.generate_thread = GenerateThread(self.engine, message, config)
        self.generate_thread.finished.connect(self.on_generation_finished)
        self.generate_thread.error.connect(self.on_generation_error)
        self.generate_thread.start()
    
    def stop_generation(self):
        """Üretimi durdur"""
        if self.generate_thread and self.generate_thread.isRunning():
            self.generate_thread.terminate()
            self.on_generation_finished("[Üretim durduruldu]")
    
    def on_generation_finished(self, text: str):
        """Üretim tamamlandı"""
        self.add_ai_message(text)
        
        self.send_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.hide()
        self.statusBar().showMessage("Hazır")
    
    def on_generation_error(self, error: str):
        """Üretim hatası"""
        self.add_system_message(f"Hata: {error}")
        
        self.send_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.hide()
        self.statusBar().showMessage(f"Hata: {error}")
    
    def add_user_message(self, message: str):
        """Kullanıcı mesajı ekle"""
        self.history.append({'role': 'user', 'content': message})
        self.chat_display.append(f'<div style="color: #a6e3a1; margin: 5px;"><b>Sen:</b></div>'
                               f'<div style="margin-left: 20px; margin-bottom: 10px;">{message}</div>')
    
    def add_ai_message(self, message: str):
        """AI mesajı ekle"""
        self.history.append({'role': 'assistant', 'content': message})
        self.chat_display.append(f'<div style="color: #89b4fa; margin: 5px;"><b>FlakeAI:</b></div>'
                               f'<div style="margin-left: 20px; margin-bottom: 10px;">{message}</div>')
    
    def add_system_message(self, message: str):
        """Sistem mesajı ekle"""
        self.chat_display.append(f'<div style="color: #f9e2af; margin: 5px;"><i>{message}</i></div>')
    
    def clear_chat(self):
        """Sohbeti temizle"""
        self.chat_display.clear()
        self.history.clear()
        self.add_system_message("Sohbet temizlendi.")
    
    def save_chat(self):
        """Sohbeti kaydet"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Sohbeti Kaydet", "",
            "Metin Dosyası (*.txt);;JSON (*.json)"
        )
        
        if file_path:
            import json
            
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.history, f, ensure_ascii=False, indent=2)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for msg in self.history:
                        role = "Sen" if msg['role'] == 'user' else "FlakeAI"
                        f.write(f"{role}: {msg['content']}\n\n")
            
            self.statusBar().showMessage(f"Sohbet kaydedildi: {file_path}")
    
    def switch_to_code_mode(self):
        """Kod moduna geç"""
        self.mode_combo.setCurrentText("Kod Yaz")
        self.input_field.setPlaceholderText("Kod açıklaması yazın...")
    
    def switch_to_chat_mode(self):
        """Sohbet moduna geç"""
        self.mode_combo.setCurrentText("Sohbet")
        self.input_field.setPlaceholderText("Mesajınızı yazın...")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = FlakeAIWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
