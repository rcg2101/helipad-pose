import numpy as np
import cv2 as cv
from pathlib import Path
from ultralytics import YOLO
from helipad_pose import estimate_pose

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
CAMERA_PATH = BASE / "camera.npz"
MODEL_PATH = ROOT / "runs" / "train" / "y26-1-db-1" / "weights" / "best.pt"
IMAGE_PATH = ROOT / "data" / "teste_beta" / "H_teste_1.png"

for path in (CAMERA_PATH, MODEL_PATH, IMAGE_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"Não encontrei: {path}")

cam = np.load(CAMERA_PATH)
K, dist = cam["K"], cam["dist"]

model = YOLO(str(MODEL_PATH))
print("tarefa:", model.task)
print("keypoints (n, dim):", model.model.kpt_shape)
print("classes:", model.names)
img = cv.imread(str(IMAGE_PATH))
if img is None:
    raise OSError(f"Não foi possível ler a imagem: {IMAGE_PATH}")
h, w = img.shape[:2]
if (w, h) != (1200, 1600):
    print(f"AVISO: imagem {w}x{h}, o K foi calibrado para 1200x1600")

res = model(img, verbose=False)[0]
if res.keypoints is None or len(res.boxes) == 0:
    raise SystemExit("Nenhuma deteção")

# deteção com maior confiança da caixa
i = int(res.boxes.conf.argmax())
pts = res.keypoints.xy[i].cpu().numpy().astype(np.float32)     # (12, 2), px da imagem original
kc = res.keypoints.conf
conf = (kc[i].cpu().numpy() if kc is not None
        else np.ones(len(pts), np.float32))
# sem visibilidade, o YOLO mete (0,0) nos pontos em falta
missing = (pts == 0).all(axis=1)
pts[missing], conf[missing] = np.nan, 0.0

print("pontos válidos:", int((conf >= 0.5).sum()), "/ 12")

out = estimate_pose(pts, conf, K, dist)
print({k: v for k, v in out.items() if k in
       ("success", "n_valid", "reproj_px", "alt_reproj_px")})

# ---- desenho: pontos numerados + eixos ----
for n, (p, c) in enumerate(zip(pts, conf), start=1):
    if np.isfinite(p).all():
        cv.circle(img, tuple(int(v) for v in p), 5,
                  (0, 255, 0) if c >= 0.5 else (0, 0, 255), -1)
        cv.putText(img, str(n), (int(p[0]) + 6, int(p[1]) - 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

if out["success"]:
    cv.drawFrameAxes(img, K, dist, out["rvec"], out["tvec"], 0.03)   # eixos de 3 cm
    print("t (m):", out["tvec"])
    print("roll, pitch, yaw (°):", out["euler_deg"])

cv.imwrite(str(BASE / "out_yolo.jpg"), img)