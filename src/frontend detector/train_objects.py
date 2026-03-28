from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

print("Iniciando o treinamento...")

results = model.train(
    data='data.yaml',         
    epochs=50,                
    imgsz=640,                
    name='treino_alimentos'   
)

print("Treinamento concluído!")