"""
FlakeAI - Web Arayüzü (Render Deploy)
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from inference.engine import FlakeInference, InferenceConfig

app = FastAPI(title="FlakeAI Agent")

engine: Optional[FlakeInference] = None


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


@app.on_event("startup")
async def startup():
    global engine
    try:
        engine = FlakeInference()
    except:
        pass


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlakeAI Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        header {
            text-align: center;
            padding: 40px 0;
        }
        h1 {
            font-size: 3em;
            background: linear-gradient(45deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle { color: #888; margin-top: 10px; }
        .chat-box {
            background: #16213e;
            border-radius: 20px;
            padding: 30px;
            margin: 20px 0;
            min-height: 400px;
            border: 1px solid #1f4068;
        }
        .message {
            margin: 15px 0;
            padding: 15px 20px;
            border-radius: 15px;
            max-width: 85%;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .user {
            background: linear-gradient(45deg, #00d4ff, #0099cc);
            margin-left: auto;
            color: white;
        }
        .ai {
            background: #1f4068;
            margin-right: auto;
        }
        .input-area {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }
        input[type="text"] {
            flex: 1;
            padding: 18px 25px;
            border-radius: 30px;
            border: 2px solid #1f4068;
            background: #16213e;
            color: white;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus { border-color: #00d4ff; }
        button {
            padding: 18px 40px;
            border-radius: 30px;
            border: none;
            background: linear-gradient(45deg, #00d4ff, #00ff88);
            color: #0f0f23;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
        }
        .footer {
            text-align: center;
            padding: 30px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>FlakeAI Agent</h1>
            <p class="subtitle">Sıfırdan eğitilen AI asistanı</p>
        </header>
        
        <div class="chat-box" id="chat">
            <div class="message ai">Merhaba! Ben FlakeAI. Size nasıl yardımcı olabilirim?</div>
        </div>
        
        <div class="input-area">
            <input type="text" id="input" placeholder="Mesajınızı yazın..." onkeypress="if(event.key==='Enter')send()">
            <button onclick="send()">Gönder</button>
        </div>
        
        <div class="footer">
            FlakeAI Agent v1.0 | Sıfırdan eğitilen AI modeli
        </div>
    </div>
    
    <script>
        let history = [];
        
        async function send() {
            const input = document.getElementById('input');
            const msg = input.value.trim();
            if (!msg) return;
            
            addMessage(msg, 'user');
            input.value = '';
            
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg, history: history})
                });
                const data = await res.json();
                addMessage(data.response, 'ai');
                history.push({user: msg, assistant: data.response});
            } catch (e) {
                addMessage('Bağlantı hatası: ' + e.message, 'ai');
            }
        }
        
        function addMessage(text, type) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.textContent = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
    </script>
</body>
</html>
"""


@app.post("/chat")
async def chat(request: ChatRequest):
    if not engine:
        return {"response": "Model henüz yüklenmedi. Lütfen bekleyin..."}
    
    try:
        response = engine.chat(request.message, request.history)
        return {"response": response}
    except Exception as e:
        return {"response": f"Hata: {str(e)}"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": engine is not None}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
