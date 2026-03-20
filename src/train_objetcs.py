from ultralytics import YOLO

# Carrega um modelo pré-treinado
model = YOLO('yolov8n.pt') 

# Inicia o treinamento usando o arquivo .yaml
results = model.train(
    data='datasets.yaml', # Aponta para o arquivo .yaml
    epochs=50,            # Número de vezes que o modelo vai processar todo o dataset
    imgsz=640,            # Redimensiona as imagens para 640x640 pixels (padrão do YOLO)
    batch=16,             # Quantidade de imagens processadas por vez
    name='modelo_alimentos' # Nome da pasta onde os resultados serão salvos
)