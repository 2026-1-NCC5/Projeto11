
import cv2
import requests
import threading
from ultralytics import YOLO
import numpy as np
import math

# --- BLOCO ADICIONADO: CALIBRAÇÃO DE TAMANHO (HOMOGRAFIA) ---
pontos_imagem_pixel = np.array([[47, 31], [555, 32], [558, 443], [60, 432]], dtype="float32")
pontos_real_cm = np.array([
    [0, 0],   
    [50, 0],  
    [50, 50], 
    [0, 50]   
], dtype="float32")

matriz_medidas, _ = cv2.findHomography(pontos_imagem_pixel, pontos_real_cm)
# -----------------------------------------------------------

API_URL = "http://localhost:8000/registrar_contagem"

print("=== SISTEMA LE - INÍCIO DE SESSÃO ===")
print("Sincronizando com o banco de dados... Aguarde.")

ID_EQUIPE = None
NOME_EQUIPE = ""

try:
    resposta = requests.get("http://localhost:8000/equipes")
    dados = resposta.json()
    
    if dados["status"] == "sucesso" and len(dados["equipes"]) > 0:
        equipes_db = dados["equipes"]
        print("\nEquipes Disponíveis:")
        
        equipes_map = {}
        for eq in equipes_db:
            print(f"{eq['id']} - {eq['nome']}")
            equipes_map[str(eq['id'])] = eq['nome']
            
        opcao = input("\nDigite o ID da equipe atual: ")
        
        if opcao in equipes_map:
            ID_EQUIPE = int(opcao)
            NOME_EQUIPE = equipes_map[opcao]
            print(f"\nSessão iniciada com sucesso para: {NOME_EQUIPE}")
        else:
            print("\nID inválido! O sistema será encerrado. Tente novamente.")
            exit()
    else:
        print("\nNenhuma equipe encontrada no banco de dados!")
        exit()
        
except Exception as e:
    print(f"\n Erro crítico: Não foi possível conectar à API local. Erro: {e}")
    exit()

print("Carregando modelo de IA... Aguarde.")
modelo = YOLO('runs/detect/treino_alimentos_final/weights/best.pt') 

camera = cv2.VideoCapture(0)
print("\nIniciando Contagem Inteligente (LE)... Pressione 'q' para sair.")

nomes_classes = {0: "Arroz", 1: "Feijao", 2: "Acucar", 3: "Macarrao", 4: "Oleo", 5: "Fuba"}
contagem_total = {} # Começa vazio para podermos adicionar os pesos dinamicamente
ids_contados = set()
linha_passagem_y = 300 

def enviar_para_api(frame_cortado, id_equipe, nome_equipe):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame_cortado)
        files = {'image': ('evidencia.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'equipe_id': id_equipe, 'equipe_nome': nome_equipe} 
        
        resposta = requests.post(API_URL, data=data, files=files)
        
        if resposta.status_code == 200:
            print(f"✅ [API/Nuvem] Salvo no Supabase: {resposta.json().get('category')} ({nome_equipe})!")
        else:
            print(f"⚠️ [API] Erro: {resposta.status_code}")
    except Exception as e:
        print(f"❌ [API] Falha ao conectar: {e}")

while True:
    sucesso, frame = camera.read()
    if not sucesso:
        break
    
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
                
                # --- CÁLCULO DE TAMANHO E PESO ---
                base_esq_pixel = np.array([[[x1, y2]]], dtype="float32")
                base_dir_pixel = np.array([[[x2, y2]]], dtype="float32")

                base_esq_cm = cv2.perspectiveTransform(base_esq_pixel, matriz_medidas)[0][0]
                base_dir_cm = cv2.perspectiveTransform(base_dir_pixel, matriz_medidas)[0][0]

                largura_cm = math.sqrt((base_dir_cm[0] - base_esq_cm[0])**2 + (base_dir_cm[1] - base_esq_cm[1])**2)
                
                nome_base = nomes_classes.get(cls_id)
                
                if nome_base:
                    # Regra do Peso: se a largura for maior que 20cm, é 5kg; se não, é 1kg.
                    if largura_cm > 20.0:
                        nome_com_peso = f"{nome_base} 5kg"
                    else:
                        nome_com_peso = f"{nome_base} 1kg"
                    
                    # Escreve os centímetros na tela do vídeo (em cima do pacote)
                    cv2.putText(frame_anotado, f"{largura_cm:.1f}cm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                # ---------------------------------------------------

                cv2.circle(frame_anotado, (centro_x, centro_y), 5, (0, 0, 255), -1)
                
                if centro_y > linha_passagem_y and obj_id not in ids_contados:
                    if nome_base:
                        if nome_com_peso not in contagem_total:
                            contagem_total[nome_com_peso] = 0
                            
                        contagem_total[nome_com_peso] += 1
                        ids_contados.add(obj_id)
                        print(f"Item Contado na tela! {nome_com_peso} ({largura_cm:.1f}cm) - ID: {obj_id}")
                        
                        thread_envio = threading.Thread(
                            target=enviar_para_api, 
                            args=(frame.copy(), ID_EQUIPE, NOME_EQUIPE)
                        )
                        thread_envio.start()

    cv2.line(frame_anotado, (0, linha_passagem_y), (frame_anotado.shape[1], linha_passagem_y), (0, 255, 0), 2)
    cv2.putText(frame_anotado, "LINHA DE CONTAGEM", (10, linha_passagem_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.rectangle(frame_anotado, (10, 10), (300, 210), (0, 0, 0), -1)
    
    cv2.putText(frame_anotado, f"ARRECADACOES ({NOME_EQUIPE}):", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
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