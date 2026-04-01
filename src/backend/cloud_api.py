from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Credenciais do Supabase ausentes no .env!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="LE Dashboard API (Cloud)", version="1.0")

# CORS super importante para o Lovable conseguir acessar sua API!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Na hora de ir pra produção, colocaremos a URL do Lovable aqui
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "online", "servico": "API do Dashboard Liderancas Empaticas"}

# ROTA 1: Pega o Ranking das Equipes (Para o gráfico de barras)
@app.get("/api/ranking")
def obter_ranking():
    try:
        resposta = supabase.table("ranking_equipes").select("*").execute()
        return {"status": "sucesso", "dados": resposta.data}
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}

# ROTA 2: Pega os Totais por Categoria (Para o gráfico de pizza)
@app.get("/api/categorias")
def obter_categorias():
    try:
        resposta = supabase.table("resumo_categorias").select("*").execute()
        return {"status": "sucesso", "dados": resposta.data}
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}

# ROTA 3: Pega o Feed em Tempo Real (Para a tabela de "Últimas Arrecadações" com fotos)
@app.get("/api/feed")
def obter_feed_recente(limite: int = 10):
    try:
        # Traz as últimas 10 contagens com o nome da equipe (fazendo um join via API)
        resposta = supabase.table("contagens_alimentos") \
            .select("id, categoria, confianca, evidencia_url, data_hora, equipes(nome)") \
            .order("data_hora", desc=True) \
            .limit(limite) \
            .execute()
        return {"status": "sucesso", "dados": resposta.data}
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}