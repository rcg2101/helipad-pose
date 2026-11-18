import time
import threading
import numpy as np
import cv2 as cv
from pathlib import Path
from ultralytics import YOLO
from collections import deque
from helipad_pose import get_candidates, track_viterbi, euler_zyx_deg

# --- Classe para capturar vídeo em background (Garante 50+ FPS sem lag) ---
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv.VideoCapture(src, cv.CAP_DSHOW)
        self.stream.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                break
            with self.lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# Carrega o modelo YOLO
BASE = Path(__file__).resolve().parent
model = YOLO(BASE.parent.parent / "models" / "best.pt")

# Inicia a câmara em background com multithreading
cap_stream = WebcamStream(0).start()
time.sleep(1.0)

w, h = 1280, 720
f = w / (2 * np.tan(np.radians(60) / 2))
K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
dist = np.zeros(5)

def put(img, text, row, color=(255, 255, 255)):
    y = 30 + 28 * row
    cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

fps = 0.0
window = deque(maxlen=10)    # Janela deslizante de 10 frames para o Viterbi
show_debug = True            # Controla a visualização dos 12 pontos e eixos 3D (tecla 'd')

while True:
    t0 = time.perf_counter()
    frame = cap_stream.read()
    if frame is None:
        break

    # YOLO otimizado com imgsz=320 para máxima velocidade de inferência
    res = model.predict(frame, imgsz=320, conf=0.25, verbose=False)[0]
    out = {"success": False}
    
    if res.keypoints is not None and len(res.boxes) > 0:
        i = int(res.boxes.conf.argmax())
        pts = res.keypoints.xy[i].cpu().numpy().astype(np.float32)
        kc = res.keypoints.conf
        conf = kc[i].cpu().numpy() if kc is not None else np.ones(12, np.float32)
        miss = (pts == 0).all(axis=1)
        pts[miss], conf[miss] = np.nan, 0.0

        # Desenha os 12 keypoints numerados no ecrã (apenas se show_debug estiver ativo)
        if show_debug:
            for n, (p, c) in enumerate(zip(pts, conf), start=1):
                if np.isfinite(p).all():
                    cv.circle(frame, tuple(int(v) for v in p), 4, (0, 255, 0) if c >= 0.5 else (0, 0, 255), -1)
                    cv.putText(frame, str(n), (int(p[0]) + 5, int(p[1]) - 5),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Adiciona os candidatos à janela deslizante (apenas uma vez por frame!)
        window.append(get_candidates(pts, conf, K, dist))
        
        # Aplica o Viterbi para suavizar a trajetória globalmente
        ch = track_viterbi(list(window), lam=0.02)
        k = ch[-1] if ch else None
        
        if k is not None and window[-1]:
            c = window[-1][k]
            
            # --- Restrição Física Rigorosa ---
            temp_euler = euler_zyx_deg(c["R"])
            roll_abs, pitch_abs = abs(temp_euler[0]), abs(temp_euler[1])
            
            # Rejeita se ultrapassar 60° de inclinação ou se Z for inferior a 10 cm
            if roll_abs > 60.0 or pitch_abs > 60.0 or c["tvec"][2] < 0.1:
                out = {"success": False}
            else:
                others = [o["reproj_px"] for j, o in enumerate(window[-1]) if j != k]
                out = {
                    "success": True, 
                    "rvec": c["rvec"], 
                    "tvec": c["tvec"],
                    "euler_deg": temp_euler, 
                    "reproj_px": c["reproj_px"],
                    "alt_reproj_px": others[0] if others else None,
                    "n_valid": int(np.sum(np.isfinite(pts).all(axis=1) & (conf >= 0.5)))
                }
    else:
        window.clear()

    # Informação textual no painel superior
    if out["success"]:
        if show_debug:
            cv.drawFrameAxes(frame, K, dist, out["rvec"], out["tvec"], 0.03)
            
        x, y, z = out["tvec"]
        roll, pitch, yaw = out["euler_deg"]
        put(frame, f"X {x*100:+7.1f} cm   Y {y*100:+7.1f} cm   Z {z*100:+7.1f} cm", 0)
        put(frame, f"Roll {roll:+6.1f}   Pitch {pitch:+6.1f}   Yaw {yaw:+6.1f} (graus)", 1)
        put(frame, f"Reproj {out['reproj_px']:.2f} px   Pontos {out['n_valid']}/12", 2)
        
        alt = out["alt_reproj_px"]
        if alt is not None and alt < 1.2 * out["reproj_px"]:   
            put(frame, "POSE AMBIGUA", 3, (0, 165, 255))
    else:
        put(frame, "H nao detetado / Fora de limite", 0, (0, 0, 255))

    # Rodapé de controlo no ecrã
    status_debug = "ON" if show_debug else "OFF"
    put(frame, f"{fps:.0f} FPS   |   [d] Pontos: {status_debug}   |   [q] Sair", 5, (0, 255, 255))
    
    cv.imshow("Pose do H", frame)
    
    # Gestão de teclas de atalho
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("d"):
        show_debug = not show_debug  # Alterna a visualização dos pontos/eixos
        
    fps = 0.9 * fps + 0.1 / max(time.perf_counter() - t0, 1e-6)

cap_stream.stop()
cv.destroyAllWindows()