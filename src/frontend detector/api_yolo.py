from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import shutil
import cv2
import os
from pathlib import Path
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração de Caminhos
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "treino_alimentos" / "weights" / "best.pt"
EVIDENCIAS_DIR = BASE_DIR / "evidencias" 

# Cria a pasta de evidências 
os.makedirs(EVIDENCIAS_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.60

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")

# Carrega o modelo treinado
model = YOLO(str(MODEL_PATH))

@app.get("/")
def root():
    return {"status": "ok", "message": "API LE - Contagem Inteligente rodando!"}

# Endpoint atualizado para receber a imagem E o nome da equipe
@app.post("/registrar_contagem")
async def registrar_contagem(
    equipe: str = Form(...), 
    image: UploadFile = File(...)
):
    timestamp_atual = datetime.now()
    str_timestamp = timestamp_atual.strftime("%Y%m%d_%H%M%S")
    
    suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        temp_path = Path(tmp.name)

    try:
        # Roda a predição do YOLO
        results = model(str(temp_path), conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Se não detectou nada, retorna status vazio
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return {
                "status": "sem_deteccao",
                "equipe": equipe,
                "timestamp": timestamp_atual.isoformat(),
                "category": "outros",
                "confidence": 0.0,
                "evidencia_path": None
            }

        # Pega a melhor detecção 
        best_box = max(results[0].boxes, key=lambda b: float(b.conf[0].item()))
        cls_id = int(best_box.cls[0].item())
        conf = float(best_box.conf[0].item())
        label = model.names[cls_id]
        
        # Gera a imagem anotada (evidência com a caixa do YOLO)
        frame_anotado = results[0].plot()
        
        # Cria um nome único para o arquivo salvo e define o caminho
        nome_arquivo = f"{equipe}_{label}_{str_timestamp}{suffix}"
        caminho_evidencia = EVIDENCIAS_DIR / nome_arquivo
        
        # Salva a evidência na pasta
        cv2.imwrite(str(caminho_evidencia), frame_anotado)

        # Retorna o payload completo exigido na documentação
        return {
            "status": "sucesso",
            "equipe": equipe,
            "timestamp": timestamp_atual.isoformat(),
            "category": str(label),
            "confidence": round(conf, 4),
            "evidencia_path": str(caminho_evidencia)
        }

    finally:
        # Limpa o arquivo temporário original
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)