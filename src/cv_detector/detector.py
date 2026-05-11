import cv2
import requests
import threading
from ultralytics import YOLO
import numpy as np
import math

pontos_imagem_pixel = np.array([[47, 31], [555, 32], [558, 443], [60, 432]], dtype="float32")
pontos_real_cm = np.array([
    [0, 0],
    [50, 0],
    [50, 50],
    [0, 50]
], dtype="float32")

matriz_medidas, _ = cv2.findHomography(pontos_imagem_pixel, pontos_real_cm)

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
    print(f"[debug] payload: {dados}")

    if dados.get("status") == "erro":
        print(f"\nAPI retornou erro: {dados.get('mensagem')}")
        exit()

    grupos = dados.get("groups") or []
    if not grupos:
        print("\nNenhum grupo encontrado no banco de dados!")
        print("Confira se SUPABASE_URL/SUPABASE_KEY no .env apontam para o projeto correto")
        print("e se a tabela public.groups tem registros.")
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
    sess_payload = resp_sess.json()
    SESSION_ID = sess_payload["session"]["id"]
    print(f"Sessão de detecção iniciada: {SESSION_ID}")
except Exception as e:
    print(f"\nErro ao iniciar sessão de detecção: {e}")
    exit()

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
ARROZ_LARGURA_CM_5KG = 20.0

contagem_total: dict[str, int] = {}
ids_contados: set[int] = set()
linha_passagem_y = 300


def peso_kg_de(categoria: str, largura_cm: float) -> float:
    if categoria == "arroz":
        return 5.0 if largura_cm > ARROZ_LARGURA_CM_5KG else 1.0
    return PESO_FIXO_KG.get(categoria, 1.0)


def formatar_peso(kg: float) -> str:
    if kg >= 1.0:
        return f"{kg:g}kg"
    return f"{int(round(kg * 1000))}g"


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
            confs = caixas.conf.cpu().tolist()

            for obj_id, cls_id, coord, conf_det in zip(ids, classes, coords, confs):
                x1, y1, x2, y2 = coord
                centro_x = int((x1 + x2) / 2)
                centro_y = int((y1 + y2) / 2)

                base_esq_pixel = np.array([[[x1, y2]]], dtype="float32")
                base_dir_pixel = np.array([[[x2, y2]]], dtype="float32")
                base_esq_cm = cv2.perspectiveTransform(base_esq_pixel, matriz_medidas)[0][0]
                base_dir_cm = cv2.perspectiveTransform(base_dir_pixel, matriz_medidas)[0][0]
                largura_cm = math.sqrt(
                    (base_dir_cm[0] - base_esq_cm[0]) ** 2
                    + (base_dir_cm[1] - base_esq_cm[1]) ** 2
                )

                categoria = nomes_classes.get(cls_id)
                weight_kg = None
                rotulo_legivel = None

                if categoria:
                    weight_kg = peso_kg_de(categoria, largura_cm)
                    rotulo_legivel = f"{categoria} {formatar_peso(weight_kg)}"
                    cv2.putText(
                        frame_anotado,
                        f"{largura_cm:.1f}cm",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

                cv2.circle(frame_anotado, (centro_x, centro_y), 5, (0, 0, 255), -1)

                if centro_y > linha_passagem_y and obj_id not in ids_contados and rotulo_legivel:
                    contagem_total[rotulo_legivel] = contagem_total.get(rotulo_legivel, 0) + 1
                    ids_contados.add(obj_id)
                    print(f"Item contado: {rotulo_legivel} ({largura_cm:.1f}cm)")

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

    cv2.line(
        frame_anotado,
        (0, linha_passagem_y),
        (frame_anotado.shape[1], linha_passagem_y),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame_anotado,
        "LINHA DE CONTAGEM",
        (10, linha_passagem_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

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
