import cv2
import numpy as np

# Variáveis globais para armazenar o estado do clique
pontos_clicados = []
imagem_exibicao = None

def capturar_clique(evento, x, y, flags, param):
    """
    Função chamada pelo OpenCV sempre que ocorre um evento de mouse na janela.
    Registra as coordenadas de cliques do botão esquerdo.
    """
    global pontos_clicados, imagem_exibicao
    
    if evento == cv2.EVENT_LBUTTONDOWN:
        if len(pontos_clicados) < 4:
            # Salva a coordenada clicada
            pontos_clicados.append([x, y])
            numero_ponto = len(pontos_clicados)
            
            print(f"Ponto {numero_ponto} registrado: (X: {x}, Y: {y})")
            
            # Feedback visual na tela: Desenha um círculo vermelho e o número do ponto
            cv2.circle(imagem_exibicao, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(imagem_exibicao, str(numero_ponto), (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Atualiza a janela com os novos desenhos
            cv2.imshow("Calibrador de Camera", imagem_exibicao)

def executar_calibracao(indice_camera=0):
    """
    Acessa a câmera, tira uma foto e abre a interface de calibração por clique.
    """
    global imagem_exibicao
    
    print(f"Acessando a câmera (Índice {indice_camera})... Aguarde.")
    camera = cv2.VideoCapture(indice_camera)
    
    # Lê alguns frames iniciais para permitir que o sensor da câmera ajuste o foco e a luz
    for _ in range(10): 
        sucesso, frame = camera.read()
    
    camera.release()
    
    if not sucesso:
        print("ERRO: Não foi possível capturar a imagem da câmera. Câmera sendo utilizada por outro programa.")
        return None
        
    imagem_exibicao = frame.copy()
    
    # Instruções no terminal
    print("\n" + "="*60)
    print("INSTRUÇÕES DE CALIBRAÇÃO (Base 30cm x 40cm com fita branca):")
    print("1. A janela com a imagem da sua câmera vai abrir.")
    print("2. Clique nos 4 cantos da fita branca da base, NESTA ORDEM:")
    print("     1º Canto Superior Esquerdo   (0cm,  0cm)")
    print("     2º Canto Superior Direito    (30cm, 0cm)")
    print("     3º Canto Inferior Direito    (30cm, 40cm)")
    print("     4º Canto Inferior Esquerdo   (0cm,  40cm)")
    print("3. Após o 4º clique, pressione 'ESPAÇO' ou 'ENTER' para finalizar.")
    print("="*60 + "\n")
    
    # Cria a janela e vincula a função de clique do mouse
    cv2.imshow("Calibrador de Camera", imagem_exibicao)
    cv2.setMouseCallback("Calibrador de Camera", capturar_clique)
    
    # Mantém a janela aberta infinitamente até que o usuário pressione uma tecla - ENTER ou ESPAÇO
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return pontos_clicados

# =====================================================================
# Programa Base
# =====================================================================
# Altere o número '0' se estiver usando uma webcam externa (ex: 1, 2)
pontos_finais = executar_calibracao(indice_camera=0)

print("\n\n" + "="*60)
# O resultado aparece no terminal.
print("             RESULTADO DA CALIBRAÇÃO")
print("="*60)

if pontos_finais and len(pontos_finais) == 4:
    print("\nCopie as 2 linhas abaixo e cole no detector.py (substituindo as existentes):\n")
    print(f'pontos_imagem_pixel = np.array({pontos_finais}, dtype="float32")')
    print('pontos_real_cm = np.array([[0, 0], [30, 0], [30, 40], [0, 40]], dtype="float32")\n')
else:
    print("\n[ERRO] Calibração incompleta. Você não clicou em 4 pontos.")
    print("Execute o script novamente.")