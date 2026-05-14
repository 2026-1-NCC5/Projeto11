import cv2
import requests
import threading
import time
import math
from collections import deque

import numpy as np
from ultralytics import YOLO

# ==============================================================================
# 1. Calibração da base (HOMOGRAFIA)
# ==============================================================================
# A base de madeira tem fita branca formando um retângulo de 30cm (largura) x
# 40cm (profundidade). Rode `calibrador.py` clicando nos 4 cantos da fita,
# nesta ordem: sup-esq, sup-dir, inf-dir, inf-esq. Cole o array gerado abaixo.
pontos_imagem_pixel = np.array([[117, 12], [535, 29], [541, 455], [89, 443]], dtype="float32")

BASE_LARGURA_CM = 30.0   # eixo X (horizontal)
BASE_PROFUND_CM = 40.0   # eixo Y (vertical, para o fundo)
pontos_real_cm = np.array([[0, 0], [30, 0], [30, 40], [0, 40]], dtype="float32")

matriz_medidas, _ = cv2.findHomography(pontos_imagem_pixel, pontos_real_cm)
matriz_inversa, _ = cv2.findHomography(pontos_real_cm, pontos_imagem_pixel)

# ==============================================================================
# 2. Parâmetros do "ficou parado dentro da base"
# ==============================================================================
ESTABILIDADE_FRAMES = 8          # frames consecutivos do mesmo id para considerar parado
ESTABILIDADE_TOL_PX = 8          # variação máxima (px) do centro nesses frames
DEDUP_DIST_CM = 5.0              # raio de duplicação espacial em cm
DEDUP_COOLDOWN_S = 3.0           # janela de duplicação (segundos)
ID_TIMEOUT_FRAMES = 30           # frames sem ver um id para considerá-lo "saiu"
CONF_MIN = 0.60                  # confiança mínima para aceitar uma detecção (0..1)
IOU_MIN = 0.50                   # IoU do NMS para evitar caixas sobrepostas

# ==============================================================================
# 3. API e sessão
# ==============================================================================
API_BASE = "http://localhost:8001"
API_EVIDENCES = f"{API_BASE}/evidences"

print("=== SISTEMA LE - INÍCIO DE SESSÃO ===")
print("Sincronizando com o banco de dados... Aguarde.")

GROUP_ID = None
GROUP_NAME = ""
SESSION_ID = None

try:
    resposta = requests.get(f"{API_BASE}/groups", timeout=10)
    print(f"[debug] GET /groups -> HTTP {resposta.status_code}")
    dados = resposta.json()

    if dados.get("status") == "erro":
        print(f"\nAPI retornou erro: {dados.get('mensagem')}")
        exit()

    grupos = dados.get("groups") or []
    if not grupos:
        print("\nNenhum grupo encontrado no banco de dados!")
        exit()
    print("\nGrupos disponíveis:")
    for i, g in enumerate(grupos, start=1):
        print(f"  [{i}] {g['name']}  (id={g['id']})")

    escolha = input("\nDigite o número do grupo: ").strip()
    if not escolha.isdigit() or not (1 <= int(escolha) <= len(grupos)):
        print("Opção inválida! Encerrando.")
        exit()

    grupo_selecionado = grupos[int(escolha) - 1]
    GROUP_ID = grupo_selecionado["id"]
    GROUP_NAME = grupo_selecionado["name"]
    print(f"\nGrupo selecionado: {GROUP_NAME}")

except Exception as e:
    print(f"\nErro crítico ao listar grupos. Erro: {e}")
    exit()

try:
    resp_sess = requests.post(
        f"{API_BASE}/sessions",
        json={"group_id": GROUP_ID},
        timeout=10,
    )
    resp_sess.raise_for_status()
    SESSION_ID = resp_sess.json()["session"]["id"]
    print(f"Sessão de detecção iniciada: {SESSION_ID}")
except Exception as e:
    print(f"\nErro ao iniciar sessão de detecção: {e}")
    exit()

# ==============================================================================
# 4. Modelo e câmera
# ==============================================================================
print("\nCarregando modelo de IA... Aguarde.")
modelo = YOLO('runs/detect/treino_alimentos_final/weights/best.pt')

camera = cv2.VideoCapture(0)
print("\nIniciando Contagem Inteligente (LE)... Pressione 'q' para sair.")

nomes_classes = {0: "arroz", 1: "feijao", 2: "acucar", 3: "macarrao", 4: "oleo", 5: "fuba"}

PESO_FIXO_KG = {
    "feijao": 1.0,
    "acucar": 1.0,
    "macarrao": 0.5,
    "oleo": 0.9,
    "fuba": 0.5,
}
ARROZ_LARGURA_CM_5KG = 20.0  # acima disso, classifica como 5kg

contagem_total: dict[str, int] = {}


def peso_kg_de(categoria: str, largura_cm: float) -> float:
    if categoria == "arroz":
        return 5.0 if largura_cm > ARROZ_LARGURA_CM_5KG else 1.0
    return PESO_FIXO_KG.get(categoria, 1.0)


def formatar_peso(kg: float) -> str:
    if kg >= 1.0:
        return f"{kg:g}kg"
    return f"{int(round(kg * 1000))}g"


def pixel_para_cm(x_px: float, y_px: float) -> tuple[float, float]:
    p = np.array([[[x_px, y_px]]], dtype="float32")
    out = cv2.perspectiveTransform(p, matriz_medidas)[0][0]
    return float(out[0]), float(out[1])


def largura_em_cm(x1: float, y1: float, x2: float, y2: float) -> float:
    base_esq = np.array([[[x1, y2]]], dtype="float32")
    base_dir = np.array([[[x2, y2]]], dtype="float32")
    a = cv2.perspectiveTransform(base_esq, matriz_medidas)[0][0]
    b = cv2.perspectiveTransform(base_dir, matriz_medidas)[0][0]
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def enviar_para_api(
    frame_cortado,
    group_id: str,
    session_id: str,
    category: str,
    confidence: float,
    weight_kg: float,
    bbox: list[int],
):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame_cortado)
        files = {'image': ('evidencia.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {
            'group_id': group_id,
            'session_id': session_id,
            'category': category,
            'confidence': f"{confidence:.4f}",
            'weight_kg': str(weight_kg),
            'bbox': ",".join(str(int(v)) for v in bbox),
        }
        resposta = requests.post(API_EVIDENCES, data=data, files=files, timeout=30)

        if resposta.status_code == 200:
            payload = resposta.json()
            status = payload.get("status")
            if status == "sucesso":
                print(f"[OK] Salvo: {payload.get('category')} {formatar_peso(weight_kg)} ({GROUP_NAME})")
            elif status == "duplicado":
                print(f"[DUP] Evidência ignorada (dedup): {payload.get('dedup_hash')}")
            else:
                print(f"[API] Resposta: {payload}")
        else:
            print(f"[API] Erro HTTP {resposta.status_code}: {resposta.text}")
    except Exception as e:
        print(f"[API] Falha ao conectar: {e}")


# ==============================================================================
# 5. Loop principal
# ==============================================================================
historico_centros: dict[int, deque] = {}
ids_contados: set[int] = set()
ultima_vez_visto: dict[int, int] = {}
contagens_recentes: list[tuple[str, float, float, float]] = []  # (cat, x_cm, y_cm, ts)
frame_idx = 0

# Polígono da base para desenhar na tela (canto da fita branca)
poligono_base_px = pontos_imagem_pixel.astype(np.int32).reshape(-1, 1, 2)

while True:
    sucesso, frame = camera.read()
    if not sucesso:
        break

    frame_idx += 1
    resultados = modelo.track(frame, persist=True, verbose=False, conf=CONF_MIN, iou=IOU_MIN)
    frame_anotado = resultados[0].plot() if resultados else frame.copy()

    # Desenha o contorno da base 30x40cm
    cv2.polylines(frame_anotado, [poligono_base_px], isClosed=True, color=(255, 255, 255), thickness=2)
    cv2.putText(frame_anotado, "BASE 30x40 cm", tuple(poligono_base_px[0][0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    ids_no_frame: set[int] = set()
    agora = time.time()

    if resultados and resultados[0].boxes is not None and resultados[0].boxes.id is not None:
        boxes = resultados[0].boxes
        ids = boxes.id.int().cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        coords = boxes.xyxy.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        for obj_id, cls_id, coord, conf_det in zip(ids, classes, coords, confs):
            if conf_det < CONF_MIN:
                continue
            x1, y1, x2, y2 = coord
            cx_px = int((x1 + x2) / 2)
            cy_px = int((y1 + y2) / 2)
            ids_no_frame.add(obj_id)
            ultima_vez_visto[obj_id] = frame_idx

            hist = historico_centros.setdefault(obj_id, deque(maxlen=ESTABILIDADE_FRAMES))
            hist.append((cx_px, cy_px))

            categoria = nomes_classes.get(cls_id)
            if categoria is None:
                continue

            largura_cm = largura_em_cm(x1, y1, x2, y2)
            weight_kg = peso_kg_de(categoria, largura_cm)
            rotulo_legivel = f"{categoria} {formatar_peso(weight_kg)}"

            # Centro do bbox em cm + se está dentro da base
            x_cm, y_cm = pixel_para_cm(cx_px, cy_px)
            dentro_base = (0.0 <= x_cm <= BASE_LARGURA_CM) and (0.0 <= y_cm <= BASE_PROFUND_CM)

            # Estabilidade: o centro variou pouco nos últimos N frames?
            estavel = False
            if len(hist) >= ESTABILIDADE_FRAMES:
                xs = [p[0] for p in hist]
                ys = [p[1] for p in hist]
                if (max(xs) - min(xs) <= ESTABILIDADE_TOL_PX
                        and max(ys) - min(ys) <= ESTABILIDADE_TOL_PX):
                    estavel = True

            # Status visual por estado
            if obj_id in ids_contados:
                cor_centro = (0, 255, 0)        # já contado → verde
                status_txt = "OK"
            elif estavel and dentro_base:
                cor_centro = (0, 200, 255)      # parado, prestes a contar → laranja
                status_txt = "PARADO"
            elif dentro_base:
                cor_centro = (0, 255, 255)      # dentro da base, mas se mexendo → amarelo
                status_txt = "MOVENDO"
            else:
                cor_centro = (0, 0, 255)        # fora da base → vermelho
                status_txt = "FORA"

            cv2.circle(frame_anotado, (cx_px, cy_px), 5, cor_centro, -1)
            cv2.putText(
                frame_anotado,
                f"{rotulo_legivel} {conf_det*100:.0f}% | {largura_cm:.1f}cm | {status_txt}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                cor_centro,
                2,
            )

            # Gatilho: parou dentro da base e ainda não foi contado
            if estavel and dentro_base and obj_id not in ids_contados:
                # Dedup espacial: já contei a mesma categoria nesse ponto recentemente?
                duplicado = any(
                    cat == categoria
                    and math.hypot(xc - x_cm, yc - y_cm) <= DEDUP_DIST_CM
                    and (agora - ts) <= DEDUP_COOLDOWN_S
                    for cat, xc, yc, ts in contagens_recentes
                )

                # Marca o id como tratado (mesmo se duplicado) para não reprocessar
                ids_contados.add(obj_id)

                if duplicado:
                    print(f"[DUP-local] {categoria} em ({x_cm:.1f},{y_cm:.1f})cm — ignorado")
                    continue

                contagens_recentes.append((categoria, x_cm, y_cm, agora))
                contagem_total[rotulo_legivel] = contagem_total.get(rotulo_legivel, 0) + 1
                print(f"Item contado: {rotulo_legivel} ({largura_cm:.1f}cm) "
                      f"em ({x_cm:.1f},{y_cm:.1f})cm")

                threading.Thread(
                    target=enviar_para_api,
                    args=(
                        frame.copy(),
                        GROUP_ID,
                        SESSION_ID,
                        categoria,
                        float(conf_det),
                        weight_kg,
                        [x1, y1, x2, y2],
                    ),
                    daemon=True,
                ).start()

    # Limpeza: ids que saíram do frame liberam slot para um próximo item
    for obj_id in list(ultima_vez_visto.keys()):
        if frame_idx - ultima_vez_visto[obj_id] > ID_TIMEOUT_FRAMES:
            ultima_vez_visto.pop(obj_id, None)
            historico_centros.pop(obj_id, None)
            ids_contados.discard(obj_id)

    contagens_recentes = [c for c in contagens_recentes if (agora - c[3]) <= DEDUP_COOLDOWN_S]

    # HUD com totais
    y_hud = 30
    for rotulo, qtd in contagem_total.items():
        cv2.putText(frame_anotado, f"{rotulo}: {qtd}", (10, y_hud),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_hud += 22

    cv2.imshow("LE - Contagem Inteligente de Alimentos", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

try:
    requests.post(f"{API_BASE}/sessions/{SESSION_ID}/end", timeout=5)
    print(f"Sessão {SESSION_ID} encerrada.")
except Exception as e:
    print(f"[API] Falha ao encerrar sessão: {e}")
