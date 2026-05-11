from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import shutil
import os
import re
import hashlib
import unicodedata
from pathlib import Path
from datetime import datetime, timezone
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
FRAMES_BUCKET = os.getenv("SUPABASE_FRAMES_BUCKET", "frames")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Erro Crítico: Credenciais do Supabase não encontradas. Verifique seu arquivo .env!")

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
EVIDENCIAS_DIR = BASE_DIR / "evidencias"
os.makedirs(EVIDENCIAS_DIR, exist_ok=True)

CATEGORIAS_VALIDAS = {"arroz", "feijao", "acucar", "macarrao", "oleo", "fuba"}


def normalizar_categoria(label: str) -> str | None:
    s = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    return s if s in CATEGORIAS_VALIDAS else None


def slugificar(texto: str) -> str:
    s = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "grupo"


def calcular_dedup_hash(category: str, bbox: list[int] | None, ts: datetime) -> str:
    bucket = int(ts.timestamp() // 5)
    if bbox:
        x1, y1, x2, y2 = bbox
        bbox_q = (x1 // 16, y1 // 16, x2 // 16, y2 // 16)
    else:
        bbox_q = (0, 0, 0, 0)
    base = f"{category}|{bbox_q}|{bucket}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


class NovoGrupo(BaseModel):
    name: str
    created_by: str | None = None


class NovaSessao(BaseModel):
    group_id: str
    started_by: str | None = None


@app.get("/")
def root():
    return {"status": "ok", "message": "API LE - Edge Gateway Rodando!"}


@app.get("/groups")
def listar_grupos():
    try:
        resposta = supabase.table("groups").select("id, name, created_by").order("name").execute()
        print(f"[/groups] {len(resposta.data or [])} grupos retornados")
        return {"status": "sucesso", "groups": resposta.data or []}
    except Exception as e:
        print(f"[/groups] ERRO: {e!r}")
        return {"status": "erro", "mensagem": str(e)}


@app.post("/groups")
def cadastrar_grupo(grupo: NovoGrupo):
    try:
        payload = {"name": grupo.name}
        if grupo.created_by:
            payload["created_by"] = grupo.created_by
        resposta = supabase.table("groups").insert(payload).execute()
        return {"status": "sucesso", "group": resposta.data[0] if resposta.data else None}
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}


@app.post("/sessions")
def iniciar_sessao(sessao: NovaSessao):
    try:
        started_by = sessao.started_by
        if not started_by:
            grupo = (
                supabase.table("groups")
                .select("created_by")
                .eq("id", sessao.group_id)
                .single()
                .execute()
            )
            started_by = grupo.data["created_by"]

        payload = {"group_id": sessao.group_id, "started_by": started_by}
        resposta = supabase.table("detection_sessions").insert(payload).execute()
        return {"status": "sucesso", "session": resposta.data[0] if resposta.data else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/end")
def encerrar_sessao(session_id: str):
    try:
        resposta = (
            supabase.table("detection_sessions")
            .update({"ended_at": datetime.now(timezone.utc).isoformat()})
            .eq("id", session_id)
            .execute()
        )
        return {"status": "sucesso", "session": resposta.data[0] if resposta.data else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evidences")
async def registrar_evidencia(
    group_id: str = Form(...),
    session_id: str = Form(...),
    category: str = Form(...),
    confidence: float = Form(...),
    weight_kg: float | None = Form(None),
    bbox: str | None = Form(None),
    image: UploadFile = File(...),
):
    categoria_norm = normalizar_categoria(category)
    if categoria_norm is None:
        raise HTTPException(status_code=400, detail=f"Categoria inválida: {category}")

    bbox_lista: list[int] | None = None
    if bbox:
        try:
            bbox_lista = [int(float(v)) for v in bbox.split(",")]
            if len(bbox_lista) != 4:
                bbox_lista = None
        except ValueError:
            bbox_lista = None

    detected_at = datetime.now(timezone.utc)
    str_timestamp = detected_at.strftime("%Y%m%d_%H%M%S_%f")
    suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        temp_path = Path(tmp.name)

    try:
        grupo_info = (
            supabase.table("groups")
            .select("name")
            .eq("id", group_id)
            .single()
            .execute()
        )
        slug = slugificar(grupo_info.data["name"]) if grupo_info.data else group_id

        nome_arquivo = f"{categoria_norm}_{str_timestamp}{suffix}"
        storage_path = f"groups/{slug}/{session_id}/{nome_arquivo}"
        caminho_local = EVIDENCIAS_DIR / nome_arquivo
        shutil.copy2(temp_path, caminho_local)

        try:
            with open(caminho_local, "rb") as f:
                supabase.storage.from_(FRAMES_BUCKET).upload(
                    path=storage_path,
                    file=f,
                    file_options={"content-type": "image/jpeg", "upsert": "true"},
                )
        except Exception as e:
            print(f"[Storage] Falha no upload: {e}")

        dedup_hash = calcular_dedup_hash(categoria_norm, bbox_lista, detected_at)
        dados_db = {
            "session_id": session_id,
            "group_id": group_id,
            "category": categoria_norm,
            "frame_url": storage_path,
            "confidence": round(float(confidence), 4),
            "detected_at": detected_at.isoformat(),
            "dedup_hash": dedup_hash,
        }
        if weight_kg is not None:
            dados_db["weight_kg"] = float(weight_kg)

        try:
            insert_resp = supabase.table("evidences").insert(dados_db).execute()
            inserted = insert_resp.data[0] if insert_resp.data else None
            print(f"[/evidences] inserido: {categoria_norm} {weight_kg}kg group={slug}")
        except Exception as e:
            msg = str(e)
            print(f"[/evidences] ERRO insert: {e!r}")
            if "dedup_hash" in msg or "duplicate" in msg.lower():
                return {"status": "duplicado", "dedup_hash": dedup_hash}
            raise

        return {
            "status": "sucesso",
            "evidence": inserted,
            "category": categoria_norm,
            "weight_kg": weight_kg,
            "frame_url": storage_path,
        }

    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
