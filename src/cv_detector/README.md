# CV Detector — Lideranças Empáticas

Pipeline de **Visão Computacional** local: captura da webcam, inferência YOLOv8 e gravação direta no **Supabase** (Postgres + Storage). Roda **sempre fora do Docker**, com Python nativo.

> Por arquitetura, o detector **não passa pelo backend FastAPI**. Ele tem um gateway HTTP próprio (`api_yolo.py`) na porta **8001** que escreve direto no Supabase. O backend só **lê** as evidências para servir o frontend.

---

## Pré-requisitos

- **Python 3.12** (recomendado via [pyenv-win](https://github.com/pyenv-win/pyenv-win) ou instalador oficial)
- Webcam acessível (índice padrão `0`)
- Acesso ao projeto Supabase (credenciais no `.env`)

---

## Setup

A partir de `src/cv_detector/`:

```bash
python -m venv venv
# Windows (Git Bash)
source venv/Scripts/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Variáveis de ambiente

Crie `src/cv_detector/.env`:

```env
SUPABASE_URL=https://<projeto>.supabase.co
SUPABASE_KEY=<service_role_key>     # service_role: necessário pra upload no bucket privado e insert sem RLS
SUPABASE_FRAMES_BUCKET=frames
```

> O bucket `frames` é privado por design. O backend gera signed URLs para o frontend visualizar.

---

## Pesos do modelo YOLO

O modelo treinado fica em `runs/detect/treino_alimentos_final/weights/best.pt`. Categorias detectadas:

| Classe YOLO | Categoria DB |
|---|---|
| 0 | arroz   |
| 1 | feijao  |
| 2 | acucar  |
| 3 | macarrao |
| 4 | oleo    |
| 5 | fuba    |

Pesos por categoria (em `detector.py`):

| Categoria | Peso |
|---|---|
| arroz | **1kg** ou **5kg** (largura > 20cm na homografia = 5kg) |
| feijão | 1kg |
| açúcar | 1kg |
| macarrão | 0.5kg |
| óleo | 0.9kg |
| fubá | 0.5kg |

---

## Rodar (fluxo da apresentação)

Em **dois terminais separados**, com a venv ativada:

**Terminal 1 — gateway HTTP local:**

```bash
uvicorn api_yolo:app --reload --host 0.0.0.0 --port 8001
```

Endpoints expostos:
- `GET /groups` — lista os grupos do banco
- `POST /sessions` — cria uma `detection_session`
- `POST /sessions/{id}/end` — encerra a sessão
- `POST /evidences` — recebe frame + metadata e grava em `evidences`

**Terminal 2 — captura via webcam:**

```bash
python detector.py
```

Fluxo no terminal:

1. Conecta no `api_yolo` e lista os grupos.
2. Você digita o número do grupo.
3. Cria uma `detection_session` automaticamente.
4. Abre janela OpenCV com a câmera. Cada pacote que cruzar a linha verde é contado e enviado.
5. Tecle **`q`** para encerrar — a sessão é fechada (`ended_at`) automaticamente.

---

## Calibrar a câmera (homografia)

A diferenciação 1kg vs 5kg do arroz usa **homografia** — sem calibrar pro setup real, todas as larguras saem absurdas e tudo vira 5kg.

```bash
python calibrador.py
```

Clique nos 4 cantos da base de captura (40×30cm, no plano da apresentação). O script imprime os 4 pontos em pixels.

Cole os valores em `detector.py`:

```python
pontos_imagem_pixel = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype="float32")
pontos_real_cm = np.array([[0, 0], [40, 0], [40, 30], [0, 30]], dtype="float32")
```

Ajuste o limiar do arroz (`ARROZ_LARGURA_CM_5KG`) se necessário (típico: ~16cm para 1kg, ~28-30cm para 5kg).

---

## Treinar / refazer o dataset

Scripts auxiliares (uso ocasional, fora do dia da apresentação):

```bash
python generate_dataset.py    # data augmentation a partir de dataset_base/
python generate_labels.py     # autolabeling dos frames
python split_dataset.py       # divide train/val
python train_objects.py       # treina YOLOv8 (gera runs/detect/treino_alimentos_*/)
```

---

## Estrutura

```
src/cv_detector/
├── api_yolo.py            # gateway FastAPI local (porta 8001)
├── detector.py            # captura webcam + tracking + envio
├── calibrador.py          # ferramenta de calibração da homografia
├── generate_dataset.py    # data augmentation
├── generate_labels.py     # autolabel YOLO
├── split_dataset.py       # split train/val
├── train_objects.py       # treino YOLOv8
├── data.yaml              # config das classes para o YOLO
├── dataset/               # dataset processado
├── dataset_base/          # imagens originais
├── runs/                  # checkpoints e métricas de treino
├── evidencias/            # frames de teste salvos localmente (ignorado pelo git)
├── requirements.txt
└── yolov8n.pt             # peso base pré-treinado
```

---

## Troubleshooting

- **`Nenhum grupo encontrado`**: verifique se a tabela `public.groups` tem linhas e se o `.env` aponta para o projeto certo.
- **`Not Found` ao chamar `/groups`**: provavelmente o `api_yolo` não está rodando ou está em outra porta. Confirme com `curl http://localhost:8001/openapi.json`.
- **Largura em cm absurdamente grande (>100cm)**: homografia descalibrada. Rode `python calibrador.py` no setup real.
- **`upload` para o bucket falha**: a key no `.env` precisa ser **service_role** (anon não tem permissão de escrita no bucket privado).
- **Câmera não abre (`cv2.VideoCapture(0)` retorna False)**: tente outros índices (`1`, `2`) ou libere o uso por outros apps.
