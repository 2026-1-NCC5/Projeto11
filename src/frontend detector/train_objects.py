from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

print("Iniciando o treinamento...")

results = model.train(
    data='data.yaml',         
    epochs=80,                
    imgsz=640,
    batch=8,           
    name='treino_alimentos'   
)

print("Treinamento concluído!")