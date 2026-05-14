import cv2
import numpy as np
import math
from ultralytics import YOLO

# ==============================================================================
# 1. Calibração da Mesa (HOMOGRAFIA)
# ==============================================================================
# Estes pontos devem ser obtidos usando o script auxiliar CALIBRADOR.PY .
# === TROQUE PELO CÓDIGO GERADO NO CALIBRADOR.PY ===
pontos_imagem_pixel = np.array([[47, 31], [555, 32], [558, 443], [60, 432]], dtype="float32")
# Coordenadas reais correspondentes em Centímetros (cm)
# Para 50cm utilize 50, para 30cm utilize 30.
pontos_real_cm = np.array([
    [0, 0],   # Canto superior esquerdo
    [50, 0],  # Canto superior direito
    [50, 50], # Canto inferior direito
    [0, 50]   # Canto inferior esquerdo
], dtype="float32")

# O OpenCV calcula a Matriz Mágica que converte Píxeis para Centímetros
matriz_medidas, _ = cv2.findHomography(pontos_imagem_pixel, pontos_real_cm)
# ==============================================================================


# ==============================================================================
# 2. Código anterior sem servidor (Iniciando modelo e câmera)
# ==============================================================================
# Carrega o seu modelo treinado (substitua 'yolov8n.pt' pelo nome do seu arquivo)
modelo = YOLO('yolov8n.pt') # Use yolov8n.pt para testar, best.pt para o seu modelo treinado

# Inicializa a câmera (webcam padrão do sistema)
camera = cv2.VideoCapture(0)

nome_janela = "Medidor de Tamanho de Pacotes"
cv2.namedWindow(nome_janela)

print("Iniciando fita métrica virtual... Pressione 'q' para sair.")
# ==============================================================================


while True:
    sucesso, frame = camera.read()
    if not sucesso:
        print("Erro ao acessar a câmera.")
        break

    # ==============================================================================
    # 3. Detectando o tamanho
    # ==============================================================================
    # A IA analisa o frame
    resultados = modelo(frame, stream=True, verbose=False)

    for r in resultados:
        # Pega as caixas de detecção
        for box in r.boxes:
            # Coordenadas da caixa delimitadora
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            nome_item = r.names[int(box.cls)]
            
            # Desenha o retângulo padrão da detecção
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # --- CÁLCULO DO TAMANHO ---
            # Apenas Largura. Vocês podem trabalhar com a altura ou área também.
            # Pegamos os dois pontos inferiores da caixa: Ponto inferior esquerdo (x1, y2) e Ponto inferior direito (x2, y2)
            base_esq_pixel = np.array([[[x1, y2]]], dtype="float32")
            base_dir_pixel = np.array([[[x2, y2]]], dtype="float32")

            # Aplica a Matriz de Homografia para converter esses píxeis em coordenadas em 'cm' do mundo real
            base_esq_cm = cv2.perspectiveTransform(base_esq_pixel, matriz_medidas)[0][0]
            base_dir_cm = cv2.perspectiveTransform(base_dir_pixel, matriz_medidas)[0][0]

            # Calcula a distância euclidiana real entre os dois pontos em centímetros
            # Fórmula: d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
            largura_cm = math.sqrt((base_dir_cm[0] - base_esq_cm[0])**2 + (base_dir_cm[1] - base_esq_cm[1])**2)
            # ---------------------------

            # --- EXIBIÇÃO DA MEDIDA ---
            # Desenha uma linha amarela grossa na base do pacote para mostrar o que está sendo medido
            cv2.line(frame, (int(x1), int(y2)), (int(x2), int(y2)), (0, 255, 255), 3)

            # Escreve o nome e a largura calculada em cm por cima do objeto
            texto_medida = f"{nome_item}: {largura_cm:.1f} cm"
            cv2.putText(frame, texto_medida, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # ---------------------------

    # ==============================================================================
    # 4. Abrindo Janela do programa
    # ==============================================================================
    # Mostra o vídeo com a medição em tempo real
    cv2.imshow(nome_janela, frame)

    # Tecla 'q' para fechar a janela
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpeza
camera.release()
cv2.destroyAllWindows()