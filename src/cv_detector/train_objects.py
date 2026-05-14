from ultralytics import YOLO

model = YOLO('yolov8n.pt')

print("Iniciando o treinamento...")

results = model.train(
    data='data.yaml',
    epochs=120,
    imgsz=640,
    batch=8,
    patience=30,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    name='treino_alimentos_final'
)

print("Treinamento concluído!")