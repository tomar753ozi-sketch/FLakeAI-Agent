"""
FlakeAI - Web Arayüzü
FastAPI tabanlı web sunucusu
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from inference.engine import FlakeInference, InferenceConfig


app = FastAPI(
    title="FlakeAI",
    description="Sıfırdan eğitilen AI modeli",
    version="1.0.0"
)

engine: Optional[FlakeInference] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


class GenerateResponse(BaseModel):
    response: str
    tokens_used: int


@app.on_event("startup")
async def startup_event():
    global engine
    try:
        engine = FlakeInference()
        print("FlakeAI model yüklendi")
    except Exception as e:
        print(f"Model yüklenemedi: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlakeAI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e1e2e 0%, #181825 100%);
            color: #cdd6f4;
            min-height: 100vh;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid #45475a;
            margin-bottom: 20px;
        }
        
        h1 {
            font-size: 2.5em;
            background: linear-gradient(45deg, #89b4fa, #a6e3a1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            background: #181825;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #45475a;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 80%;
        }
        
        .user-message {
            background: #313244;
            margin-left: auto;
            border-bottom-right-radius: 0;
        }
        
        .ai-message {
            background: #45475a;
            margin-right: auto;
            border-bottom-left-radius: 0;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
        }
        
        input[type="text"] {
            flex: 1;
            background: #313244;
            border: 1px solid #45475a;
            border-radius: 10px;
            padding: 15px 20px;
            color: #cdd6f4;
            font-size: 16px;
            outline: none;
        }
        
        input[type="text"]:focus {
            border-color: #89b4fa;
        }
        
        button {
            background: linear-gradient(45deg, #89b4fa, #74c7ec);
            border: none;
            border-radius: 10px;
            padding: 15px 30px;
            color: #1e1e2e;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(137, 180, 250, 0.3);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            background: #45475a;
            color: #6c7086;
            cursor: not-allowed;
        }
        
        .settings {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            padding: 15px;
            background: #313244;
            border-radius: 10px;
        }
        
        .setting-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        label {
            color: #a6adc8;
        }
        
        input[type="range"] {
            width: 100px;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #6c7086;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>FlakeAI</h1>
            <p>Sıfırdan eğitilen AI modeli</p>
        </header>
        
        <div class="settings">
            <div class="setting-item">
                <label>Sıcaklık:</label>
                <input type="range" id="temperature" min="0.1" max="2" step="0.1" value="0.8">
                <span id="temp-value">0.8</span>
            </div>
            <div class="setting-item">
                <label>Max Token:</label>
                <input type="number" id="max-tokens" value="512" min="64" max="4096" style="width: 80px;">
            </div>
            <div class="setting-item">
                <label>Top-k:</label>
                <input type="number" id="top-k" value="50" min="1" max="100" style="width: 60px;">
            </div>
        </div>
        
        <div class="chat-container" id="chat">
            <div class="message ai-message">
                Merhaba! Ben FlakeAI. Size nasıl yardımcı olabilirim?
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" id="user-input" placeholder="Mesajınızı yazın..." autofocus>
            <button id="send-btn" onclick="sendMessage()">Gönder</button>
        </div>
    </div>
    
    <script>
        const chat = document.getElementById('chat');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const tempSlider = document.getElementById('temperature');
        const tempValue = document.getElementById('temp-value');
        const maxTokens = document.getElementById('max-tokens');
        const topK = document.getElementById('top-k');
        
        let history = [];
        
        tempSlider.oninput = function() {
            tempValue.textContent = this.value;
        }
        
        userInput.onkeypress = function(e) {
            if (e.key === 'Enter') sendMessage();
        }
        
        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            userInput.value = '';
            sendBtn.disabled = true;
            
            addLoading();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: message,
                        history: history
                    })
                });
                
                const data = await response.json();
                removeLoading();
                addMessage(data.response, 'ai');
                history.push({user: message, assistant: data.response});
            } catch (error) {
                removeLoading();
                addMessage('Hata oluştu: ' + error.message, 'ai');
            }
            
            sendBtn.disabled = false;
            userInput.focus();
        }
        
        function addMessage(text, sender) {
            const div = document.createElement('div');
            div.className = 'message ' + sender + '-message';
            div.textContent = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        function addLoading() {
            const div = document.createElement('div');
            div.className = 'loading';
            div.id = 'loading';
            div.textContent = 'Düşünüyorum...';
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        function removeLoading() {
            const loading = document.getElementById('loading');
            if (loading) loading.remove();
        }
    </script>
</body>
</html>
"""


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Metin üret"""
    if not engine:
        raise HTTPException(status_code=503, detail="Model yüklenmedi")
    
    try:
        config = {
            'max_new_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_k': request.top_k,
            'top_p': request.top_p
        }
        
        response = engine.generate(request.prompt, **config)
        
        return GenerateResponse(
            response=response,
            tokens_used=len(response.split())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """Sohbet modu"""
    if not engine:
        raise HTTPException(status_code=503, detail="Model yüklenmedi")
    
    try:
        response = engine.chat(request.message, request.history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Sağlık kontrolü"""
    return {
        "status": "ok",
        "model_loaded": engine is not None
    }


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Sunucuyu çalıştır"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FlakeAI Web Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host')
    parser.add_argument('--port', type=int, default=8080, help='Port')
    
    args = parser.parse_args()
    run_server(args.host, args.port)
