import cv2
import requests
import threading
from ultralytics import YOLO

# --- CONFIGURAÇÕES DA API ---
API_URL = "http://localhost:8000/registrar_contagem"
NOME_EQUIPE = "Equipe Alpha" # Coloque o nome real da equipe arrecadadora aqui

# Carrega o modelo treinado
modelo = YOLO('runs/detect/treino_alimentos/weights/best.pt') 

camera = cv2.VideoCapture(0)
print("Iniciando Contagem Inteligente (LE)... Pressione 'q' para sair.")

# Classes alinhadas (mantendo o Macarrão separado, conforme sua preferência)
nomes_classes = {0: "Arroz", 1: "Feijao", 2: "Acucar", 3: "Macarrao"}
contagem_total = {"Arroz": 0, "Feijao": 0, "Acucar": 0, "Macarrao": 0}
ids_contados = set()
linha_passagem_y = 300 

# --- FUNÇÃO PARA ENVIAR DADOS EM SEGUNDO PLANO ---
def enviar_para_api(frame_cortado, equipe):
    try:
        # Codifica a imagem da câmera em formato JPG na memória
        _, img_encoded = cv2.imencode('.jpg', frame_cortado)
        
        # Prepara os arquivos e o formulário para enviar ao FastAPI
        files = {'image': ('evidencia.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'equipe': equipe}
        
        # Dispara o POST para a API
        resposta = requests.post(API_URL, data=data, files=files)
        
        if resposta.status_code == 200:
            print(f"✅ [API] Salvo na nuvem: {resposta.json().get('category')}!")
        else:
            print(f"⚠️ [API] Erro: {resposta.status_code}")
    except Exception as e:
        print(f"❌ [API] Falha ao conectar com o servidor: {e}")


while True:
    sucesso, frame = camera.read()
    if not sucesso:
        print("Erro ao acessar a câmera!")
        break
    
    # Rastreamento
    resultados = modelo.track(frame, persist=True, verbose=False)
    
    for resultado in resultados:
        frame_anotado = resultado.plot() 
        caixas = resultado.boxes
        
        if caixas.id is not None:
            ids = caixas.id.int().cpu().tolist()
            classes = caixas.cls.int().cpu().tolist()
            coords = caixas.xyxy.int().cpu().tolist()
            
            for obj_id, cls_id, coord in zip(ids, classes, coords):
                x1, y1, x2, y2 = coord
                centro_x = int((x1 + x2) / 2)
                centro_y = int((y1 + y2) / 2)
                
                cv2.circle(frame_anotado, (centro_x, centro_y), 5, (0, 0, 255), -1)
                
                # --- LÓGICA DE CONTAGEM E ENVIO ---
                if centro_y > linha_passagem_y and obj_id not in ids_contados:
                    nome_categoria = nomes_classes.get(cls_id)
                    
                    if nome_categoria:
                        contagem_total[nome_categoria] += 1
                        ids_contados.add(obj_id)
                        print(f"Item Contado na tela! {nome_categoria} (ID: {obj_id})")
                        
                        # Inicia uma "Thread" para enviar a imagem original limpa para a API validar e salvar
                        thread_envio = threading.Thread(
                            target=enviar_para_api, 
                            args=(frame.copy(), NOME_EQUIPE)
                        )
                        thread_envio.start()

    # --- DESENHO DA INTERFACE NA TELA ---
    cv2.line(frame_anotado, (0, linha_passagem_y), (frame_anotado.shape[1], linha_passagem_y), (0, 255, 0), 2)
    cv2.putText(frame_anotado, "LINHA DE CONTAGEM", (10, linha_passagem_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.rectangle(frame_anotado, (10, 10), (300, 150), (0, 0, 0), -1)
    cv2.putText(frame_anotado, "TOTAL ARRECADADO (LE):", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_pos = 60
    for categoria, quantidade in contagem_total.items():
        texto = f"{categoria}: {quantidade}"
        cv2.putText(frame_anotado, texto, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25

    cv2.imshow("LE - Contagem Inteligente de Alimentos", frame_anotado)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()