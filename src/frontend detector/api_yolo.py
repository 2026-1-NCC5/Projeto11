from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import shutil
from pathlib import Path

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
CONFIDENCE_THRESHOLD = 0.60

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))


@app.get("/")
def root():
    return {"status": "ok", "message": "API YOLO da camera_ai rodando"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        temp_path = Path(tmp.name)

    try:
        results = model(str(temp_path), conf=CONFIDENCE_THRESHOLD, verbose=False)

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return {
                "category": "outros",
                "confidence": 0.0,
                "image_path": None
            }

        best_box = max(results[0].boxes, key=lambda b: float(b.conf[0].item()))
        cls_id = int(best_box.cls[0].item())
        conf = float(best_box.conf[0].item())
        label = model.names[cls_id]

        return {
            "category": str(label),
            "confidence": round(conf, 4),
            "image_path": None
        }

    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)