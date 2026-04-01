import cv2
import requests
import threading
from ultralytics import YOLO

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
modelo = YOLO('runs/detect/treino_alimentos/weights/best.pt') 

camera = cv2.VideoCapture(0)
print("\nIniciando Contagem Inteligente (LE)... Pressione 'q' para sair.")

nomes_classes = {0: "Arroz", 1: "Feijao", 2: "Acucar", 3: "Macarrao", 4: "Oleo", 5: "Fuba"}
contagem_total = {nome: 0 for nome in nomes_classes.values()}
ids_contados = set()
linha_passagem_y = 300 

# Recebe id e nome da equipe
def enviar_para_api(frame_cortado, id_equipe, nome_equipe):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame_cortado)
        files = {'image': ('evidencia.jpg', img_encoded.tobytes(), 'image/jpeg')}
        # Envia as duas informações para o Form da FastAPI
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
                
                cv2.circle(frame_anotado, (centro_x, centro_y), 5, (0, 0, 255), -1)
                
                if centro_y > linha_passagem_y and obj_id not in ids_contados:
                    nome_categoria = nomes_classes.get(cls_id)
                    
                    if nome_categoria:
                        contagem_total[nome_categoria] += 1
                        ids_contados.add(obj_id)
                        print(f"Item Contado na tela! {nome_categoria} (ID: {obj_id}) - {NOME_EQUIPE}")
                        
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