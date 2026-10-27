import numpy as np
import cv2 as cv
from pathlib import Path

BASE = Path(__file__).resolve().parent
cam = np.load(BASE / "camera.npz")
K, dist = cam["K"], cam["dist"]

# ---------- Modelo 3D do H (MEDIDAS PROVISÓRIAS, em metros) ----------
W = 0.066    # largura total do H (6,6 cm)
HT = 0.0825  # altura total do H (8,25 cm)
T = 0.0165   # espessura de cada barra vertical (1,65 cm)
C = 0.0165   # espessura da barra transversal (1,65 cm, centrada)

def h_model(W=W, HT=HT, T=T, C=C):
    """12 vértices do H, origem no centro, X para a direita, Y para baixo, Z=0.
    Ordem: a partir do canto superior esquerdo, no sentido horário."""
    w, h, c = W / 2, HT / 2, C / 2
    pts = [(-w, -h), (-w + T, -h), (-w + T, -c), (w - T, -c),
           (w - T, -h), (w, -h), (w, h), (w - T, h),
           (w - T, c), (-w + T, c), (-w + T, h), (-w, h)]
    return np.array([(x, y, 0.0) for x, y in pts], np.float32)

# ---------- Funções de erro ----------
def rot_error_deg(R_gt, R_est):
    """Ângulo da rotação relativa (robusto, sem Euler)."""
    cosang = (np.trace(R_gt.T @ R_est) - 1) / 2
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

def estimate_pose(pts2d, pts3d, K, dist):
    """IPPE devolve 2 soluções (ambiguidade); escolhe a de menor erro de reprojeção."""
    n, rvecs, tvecs, errs = cv.solvePnPGeneric(
    obj, pts2d.astype(np.float32), K, dist, flags=cv.SOLVEPNP_IPPE)
    rot_errs = [rot_error_deg(R_gt, cv.Rodrigues(r)[0]) for r in rvecs]
    if n == 0:
        return None
    best = int(np.argmin(errs))
    return rvecs[best], tvecs[best], float(np.ravel(errs)[best])

def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

# ---------- Teste ----------
rng = np.random.default_rng(0)
obj = h_model()

for zmin, zmax in [(0.6, 1.5), (0.25, 0.8)]:
    print(f"\n=== distância z em [{zmin}, {zmax}] m ===")
    for sigma in [0.0, 0.5, 1.0]:
        t_err, r_err, fail, rescued = [], [], 0, 0
        for _ in range(200):
            R_gt = (Rz(rng.uniform(-np.pi, np.pi))
                    @ Rx(np.radians(rng.uniform(-40, 40)))
                    @ Ry(np.radians(rng.uniform(-40, 40))))
            t_gt = np.array([rng.uniform(-0.1, 0.1),
                             rng.uniform(-0.1, 0.1),
                             rng.uniform(zmin, zmax)])
            rvec_gt, _ = cv.Rodrigues(R_gt)

            proj, _ = cv.projectPoints(obj, rvec_gt, t_gt, K, dist)
            pts2d = (proj.reshape(-1, 2)
                     + rng.normal(0, sigma, (12, 2))).astype(np.float32)

            n, rvecs, tvecs, errs = cv.solvePnPGeneric(
                obj, pts2d, K, dist, flags=cv.SOLVEPNP_IPPE)
            if n == 0:
                continue
            best = int(np.argmin(errs))
            R_est, _ = cv.Rodrigues(rvecs[best])
            e_rot = rot_error_deg(R_gt, R_est)
            r_err.append(e_rot)
            t_err.append(np.linalg.norm(tvecs[best].ravel() - t_gt) * 100)

            if e_rot > 10:                      # falha de ambiguidade
                fail += 1
                alt = [rot_error_deg(R_gt, cv.Rodrigues(r)[0]) for r in rvecs]
                if min(alt) < 10:               # a outra solução estava certa
                    rescued += 1

        r_err = np.array(r_err); t_err = np.array(t_err)
        print(f"ruído {sigma:.1f} px | rot: mediana {np.median(r_err):.2f}°, "
              f"p95 {np.percentile(r_err, 95):.2f}° | "
              f"t: mediana {np.median(t_err):.2f} cm | "
              f"falhas>10°: {fail}/200 | outra solução certa: {rescued}/{fail}")