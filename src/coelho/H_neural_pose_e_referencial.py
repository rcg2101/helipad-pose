import time
import numpy as np
import cv2 as cv
from pathlib import Path
from ultralytics import YOLO
from helipad_pose import estimate_pose

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
MODEL_PATH = ROOT / "runs" / "train" / "y26-1-db-1" / "weights" / "best.pt"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Não encontrei os pesos do modelo: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# K APROXIMADO (60° de campo de visão): as distâncias não são fiáveis
f = w / (2 * np.tan(np.radians(60) / 2))
K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
dist = np.zeros(5)

def put(img, text, row, color=(255, 255, 255)):
    y = 30 + 28 * row
    cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)   # contorno
    cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

fps = 0.0
while True:
    t0 = time.perf_counter()
    ok, frame = cap.read()
    if not ok:
        break

    res = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
    out = {"success": False}
    if res.keypoints is not None and len(res.boxes) > 0:
        i = int(res.boxes.conf.argmax())
        pts = res.keypoints.xy[i].cpu().numpy().astype(np.float32)
        kc = res.keypoints.conf
        conf = kc[i].cpu().numpy() if kc is not None else np.ones(12, np.float32)
        miss = (pts == 0).all(axis=1)
        pts[miss], conf[miss] = np.nan, 0.0
        out = estimate_pose(pts, conf, K, dist)

    if out["success"]:
        cv.drawFrameAxes(frame, K, dist, out["rvec"], out["tvec"], 0.03)
        x, y, z = out["tvec"]
        roll, pitch, yaw = out["euler_deg"]
        put(frame, f"X {x*100:+7.1f} cm   Y {y*100:+7.1f} cm   Z {z*100:+7.1f} cm", 0)
        put(frame, f"Roll {roll:+6.1f}   Pitch {pitch:+6.1f}   Yaw {yaw:+6.1f} (graus)", 1)
        put(frame, f"Reproj {out['reproj_px']:.2f} px   Pontos {out['n_valid']}/12", 2)
        alt = out["alt_reproj_px"]
        if alt is not None and alt < 1.2 * out["reproj_px"]:   # limiar heurístico
            put(frame, "POSE AMBIGUA", 3, (0, 165, 255))
    else:
        put(frame, "H nao detetado", 0, (0, 0, 255))

    put(frame, f"{fps:.0f} FPS   (K aproximado)   [q] sair", 5, (0, 255, 255))
    cv.imshow("Pose do H", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    fps = 0.9 * fps + 0.1 / max(time.perf_counter() - t0, 1e-6)

cap.release()
cv.destroyAllWindows()  