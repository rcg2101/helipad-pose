import numpy as np
import cv2 as cv
from pathlib import Path
from ultralytics import YOLO
from helipad_pose import estimate_pose

# Define os caminhos e carrega os parâmetros da câmara calibrada
BASE = Path(__file__).resolve().parent
cam = np.load(BASE / "camera.npz")
K, dist = cam["K"], cam["dist"]

# Caminho para o modelo YOLO treinado e para a imagem de teste
MODEL = r"C:\Users\Asus\helipad-pose\models\best.pt"
IMG = BASE.parent.parent / "data" / "teste" / "foto1.jpeg"   # ADAPTAR se necessário

# Garante que os ficheiros existem antes de continuar
for p in (MODEL, IMG):
    if not Path(p).exists():
        raise FileNotFoundError(f"Não encontrei: {p}")

# Carrega o modelo YOLO e mostra informações básicas na consola
model = YOLO(MODEL)
print("tarefa:", model.task)
print("keypoints (n, dim):", model.model.kpt_shape)
print("classes:", model.names)

# Lê a imagem de teste e verifica se a resolução coincide com a da calibração
img = cv.imread(str(IMG))
h, w = img.shape[:2]
if (w, h) != (1200, 1600):
    print(f"AVISO: imagem {w}x{h}, o K foi calibrado para 1200x1600")

# Executa a deteção com o YOLO na imagem
res = model(img, verbose=False)[0]
if res.keypoints is None or len(res.boxes) == 0:
    raise SystemExit("Nenhuma deteção")

# Seleciona apenas a caixa (bounding box) com maior grau de confiança
i = int(res.boxes.conf.argmax())
pts = res.keypoints.xy[i].cpu().numpy().astype(np.float32)     # Coordenadas 2D dos 12 cantos
kc = res.keypoints.conf
conf = (kc[i].cpu().numpy() if kc is not None
        else np.ones(len(pts), np.float32))

# O YOLO coloca (0,0) nos pontos que não consegue ver; o script converte isso para NaN
missing = (pts == 0).all(axis=1)
pts[missing], conf[missing] = np.nan, 0.0

print("pontos válidos:", int((conf >= 0.5).sum()), "/ 12")

# Estima a pose 3D usando as deteções do YOLO e os parâmetros da câmara
out = estimate_pose(pts, conf, K, dist)
print({k: v for k, v in out.items() if k in
       ("success", "n_valid", "reproj_px", "alt_reproj_px")})

# ---- Desenho na imagem: marca os pontos detetados e os eixos 3D ----
for n, (p, c) in enumerate(zip(pts, conf), start=1):
    if np.isfinite(p).all():
        # Círculo verde se a confiança for alta (>=0.5), vermelho se for baixa (<0.5)
        cv.circle(img, tuple(int(v) for v in p), 5,
                  (0, 255, 0) if c >= 0.5 else (0, 0, 255), -1)
        # Escreve o número do canto ao lado
        cv.putText(img, str(n), (int(p[0]) + 6, int(p[1]) - 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# Se a estimativa de pose teve sucesso, desenha os eixos 3D (X, Y, Z) sobre o alvo
if out["success"]:
    cv.drawFrameAxes(img, K, dist, out["rvec"], out["tvec"], 0.03)   # Eixos com 3 cm de comprimento
    print("t (m):", out["tvec"])
    print("roll, pitch, yaw (°):", out["euler_deg"])

# Guarda a imagem final com as anotações visuais na pasta do script
cv.imwrite(str(BASE / "out_yolo.jpg"), img)