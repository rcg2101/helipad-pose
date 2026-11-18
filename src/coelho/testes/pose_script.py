import numpy as np
import cv2 as cv
from pathlib import Path

# Descobrir onde este script está guardado para carregar os ficheiros ao lado
BASE = Path(__file__).resolve().parent
cam = np.load(BASE / "camera.npz")
K, dist = cam["K"], cam["dist"] # Parâmetros da câmara guardados na calibração

# ---------- Dimensões reais do objeto em forma de "H" (em metros) ----------
W = 0.066    # largura total do H
HT = 0.0825  # altura total do H
T = 0.0165   # espessura das barras verticais
C = 0.0165   # espessura da barra horizontal do meio

def h_model(W=W, HT=HT, T=T, C=C):
    """Cria os 12 pontos 3D do H num plano (Z=0).
    Começa no canto superior esquerdo e anda no sentido horário."""
    w, h, c = W / 2, HT / 2, C / 2
    pts = [(-w, -h), (-w + T, -h), (-w + T, -c), (w - T, -c),
           (w - T, -h), (w, -h), (w, h), (w - T, h),
           (w - T, c), (-w + T, c), (-w + T, h), (-w, h)]
    return np.array([(x, y, 0.0) for x, y in pts], np.float32)

# ---------- Funções matemáticas úteis ----------
def rot_error_deg(R_gt, R_est):
    """Calcula o erro de rotação em graus entre a pose real e a estimada."""
    cosang = (np.trace(R_gt.T @ R_est) - 1) / 2
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

# ---------- Teste de simulação (Monte Carlo) ----------
rng = np.random.default_rng(0) # Gerador de números aleatórios consistente
obj = h_model()

# Testar para dois intervalos diferentes de distância (z) da câmara ao objeto
for zmin, zmax in [(0.6, 1.5), (0.25, 0.8)]:
    print(f"\n=== Distância z entre {zmin}m e {zmax}m ===")
    
    # Testar com diferentes quantidades de ruído nos pixéis da imagem
    for sigma in [0.0, 0.5, 1.0]:
        t_err, r_err, fail, rescued = [], [], 0, 0
        
        # Repetir o teste 200 vezes para obter estatísticas fiáveis
        for _ in range(200):
            # Gerar uma rotação e translação aleatória no espaço 3D
            R_gt = (Rz(rng.uniform(-np.pi, np.pi))
                    @ Rx(np.radians(rng.uniform(-40, 40)))
                    @ Ry(np.radians(rng.uniform(-40, 40))))
            t_gt = np.array([rng.uniform(-0.1, 0.1),
                             rng.uniform(-0.1, 0.1),
                             rng.uniform(zmin, zmax)])
            rvec_gt, _ = cv.Rodrigues(R_gt)

            # Projetar os pontos 3D para 2D na câmara e adicionar ruído artificial
            proj, _ = cv.projectPoints(obj, rvec_gt, t_gt, K, dist)
            pts2d = (proj.reshape(-1, 2)
                     + rng.normal(0, sigma, (12, 2))).astype(np.float32)

            # Estimar a pose com o IPPE (ótimo para alvos planos com ambiguidades)
            n, rvecs, tvecs, errs = cv.solvePnPGeneric(
                obj, pts2d, K, dist, flags=cv.SOLVEPNP_IPPE)
            if n == 0:
                continue
                
            # Selecionar a solução com o menor erro de projeção
            best = int(np.argmin(errs))
            R_est, _ = cv.Rodrigues(rvecs[best])
            
            # Guardar os erros calculados de rotação e translação
            e_rot = rot_error_deg(R_gt, R_est)
            r_err.append(e_rot)
            t_err.append(np.linalg.norm(tvecs[best].ravel() - t_gt) * 100)

            # Detetar se houve falha por ambiguidade (erro de rotação maior que 10 graus)
            if e_rot > 10:                      
                fail += 1
                alt = [rot_error_deg(R_gt, cv.Rodrigues(r)[0]) for r in rvecs]
                if min(alt) < 10:               # A outra solução calculada estava correta
                    rescued += 1

        r_err = np.array(r_err); t_err = np.array(t_err)
        print(f"Ruído {sigma:.1f} px | Rot: mediana {np.median(r_err):.2f}°, "
              f"p95 {np.percentile(r_err, 95):.2f}° | "
              f"Trans: mediana {np.median(t_err):.2f} cm | "
              f"Falhas > 10°: {fail}/200 | Outra solução certa: {rescued}/{fail}")