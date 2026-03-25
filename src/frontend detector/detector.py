import cv2
from ultralytics import YOLO

modelo = YOLO('yolov8n.pt') 

camera = cv2.VideoCapture(0)
print("Iniciando Contagem Inteligente de Alimentos (LE)... Pressione 'q' para sair.")

# Classes alinhadas com  arquivo datasets.yaml
nomes_classes = {0: "Arroz", 1: "Feijao", 2: "acucar"}

# Dicionário para armazenar o TOTAL acumulado
contagem_total = {"Arroz": 0, "Feijao": 0, "acucar": 0}

# Conjunto (Set) para armazenar os IDs únicos dos pacotes que já foram contados
ids_contados = set()

# Posição da "Linha de Passagem" (eixo Y)
linha_passagem_y = 300 

while True:
    sucesso, frame = camera.read()
    if not sucesso:
        print("Erro ao acessar a câmera!")
        break
    
    # Rastreamento dos pacotes para evitar duplicidade
    resultados = modelo.track(frame, persist=True, verbose=False)
    
    for resultado in resultados:
        frame_anotado = resultado.plot() # Desenha as caixas do YOLO
        caixas = resultado.boxes
        
        # Verifica se o rastreador conseguiu atribuir IDs
        if caixas.id is not None:
            ids = caixas.id.int().cpu().tolist()
            classes = caixas.cls.int().cpu().tolist()
            coords = caixas.xyxy.int().cpu().tolist()
            
            for obj_id, cls_id, coord in zip(ids, classes, coords):
                x1, y1, x2, y2 = coord
                
                # Centro geométrico do pacote
                centro_x = int((x1 + x2) / 2)
                centro_y = int((y1 + y2) / 2)
                
                cv2.circle(frame_anotado, (centro_x, centro_y), 5, (0, 0, 255), -1)
                
                # LÓGICA DE CONTAGEM (Cruzar a linha + ID inédito)
                if centro_y > linha_passagem_y and obj_id not in ids_contados:
                    
                    # Busca o nome da classe (0, 1 ou 2)
                    nome_categoria = nomes_classes.get(cls_id)
                    
                    # Se for uma das 3 classes conhecidas, adiciona ao placar
                    if nome_categoria:
                        contagem_total[nome_categoria] += 1
                        ids_contados.add(obj_id)
                        print(f"Item Contado! {nome_categoria} (ID: {obj_id}) - Total: {contagem_total[nome_categoria]}")

    # --- DESENHO DA INTERFACE ---
    
    cv2.line(frame_anotado, (0, linha_passagem_y), (frame_anotado.shape[1], linha_passagem_y), (0, 255, 0), 2)
    cv2.putText(frame_anotado, "LINHA DE CONTAGEM", (10, linha_passagem_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.rectangle(frame_anotado, (10, 10), (300, 130), (0, 0, 0), -1)
    cv2.putText(frame_anotado, "TOTAL ARRECADADO (LE):", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_pos = 60
    # Mostra a contagem acumulada (Arroz, Feijão, Macarrão)
    for categoria, quantidade in contagem_total.items():
        texto = f"{categoria}: {quantidade}"
        cv2.putText(frame_anotado, texto, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25

    cv2.imshow("LE - Contagem Inteligente de Alimentos", frame_anotado)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()