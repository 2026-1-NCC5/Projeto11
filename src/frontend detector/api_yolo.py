from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import shutil
import cv2
import os
from pathlib import Path
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv 

# Carrega as variáveis do arquivo .env para a memória
load_dotenv()

# --- CREDENCIAIS DO SUPABASE 
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Erro Crítico: Credenciais do Supabase não encontradas. Verifique seu arquivo .env!")

# Inicializa a conexão com o banco
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "treino_alimentos" / "weights" / "best.pt"
EVIDENCIAS_DIR = BASE_DIR / "evidencias" 
os.makedirs(EVIDENCIAS_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.60

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

@app.get("/")
def root():
    return {"status": "ok", "message": "API LE - Edge Gateway Rodando!"}

@app.get("/equipes")
def listar_equipes():
    try:
        resposta = supabase.table("equipes").select("id, nome").order("id").execute()
        return {"status": "sucesso", "equipes": resposta.data}
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}

# Recebe o ID e o Nome
@app.post("/registrar_contagem")
async def registrar_contagem(
    equipe_id: int = Form(...),
    equipe_nome: str = Form(...), 
    image: UploadFile = File(...)
):
    timestamp_atual = datetime.now()
    str_timestamp = timestamp_atual.strftime("%Y%m%d_%H%M%S")
    suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        temp_path = Path(tmp.name)

    try:
        results = model(str(temp_path), conf=CONFIDENCE_THRESHOLD, verbose=False)

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return {"status": "sem_deteccao"}

        best_box = max(results[0].boxes, key=lambda b: float(b.conf[0].item()))
        cls_id = int(best_box.cls[0].item())
        conf = float(best_box.conf[0].item())
        label = model.names[cls_id]
        
        frame_anotado = results[0].plot()
        nome_arquivo = f"{equipe_id}_{label}_{str_timestamp}{suffix}"
        caminho_evidencia = EVIDENCIAS_DIR / nome_arquivo
        
        cv2.imwrite(str(caminho_evidencia), frame_anotado)

        # 1. UPLOAD PARA O BUCKET 'evidencias' NO SUPABASE
        url_publica = None
        try:
            with open(caminho_evidencia, "rb") as f:
                supabase.storage.from_("evidencias").upload(
                    path=nome_arquivo, 
                    file=f, 
                    file_options={"content-type": "image/jpeg"}
                )
            # Pega o link público da imagem recém salva
            url_publica = supabase.storage.from_("evidencias").get_public_url(nome_arquivo)
        except Exception as e:
            print(f"Erro ao fazer upload para o Storage: {e}")

        # 2. SALVAR NO BANCO DE DADOS POSTGRESQL (Tabela contagens_alimentos)
        try:
            dados_db = {
                "equipe_id": equipe_id,
                "categoria": str(label),
                "confianca": round(conf, 4),
                "evidencia_url": url_publica
            }
            supabase.table("contagens_alimentos").insert(dados_db).execute()
        except Exception as e:
            print(f"Erro ao salvar no banco de dados: {e}")

        return {
            "status": "sucesso",
            "equipe": equipe_nome,
            "category": str(label),
            "evidencia_url": url_publica
        }

    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)