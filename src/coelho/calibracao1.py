import numpy as np
import cv2 as cv
import glob
from pathlib import Path

# Critério de paragem do refinamento sub-pixel
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ADAPTAR: cantos INTERNOS (colunas, linhas) e lado do quadrado em metros
PATTERN = (6, 8)
SQUARE = 0.025

# Pontos 3D do tabuleiro: (0,0,0), (1,0,0), ... (6,5,0)
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
objp *= SQUARE   # ADAPTAÇÃO 1: em metros em vez de "unidades de quadrado"

objpoints = []   # pontos 3D no mundo real
imgpoints = []   # pontos 2D na imagem

BASE = Path(__file__).resolve().parent        # ...\src\coelho
CALIB_DIR = BASE.parent.parent / "data" / "calib"   # ...\helipad-pose\data\calib

images = []
for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
    images += glob.glob(str(CALIB_DIR / ext))

print("pasta de calibração:", CALIB_DIR)
print("imagens encontradas:", len(images))

if not images:
    raise FileNotFoundError(f"Nenhuma imagem encontrada em {CALIB_DIR}")

import os
print("pasta atual:", os.getcwd())
print("imagens encontradas:", len(images))

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, PATTERN, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Desenha os cantos para confirmares visualmente
        cv.drawChessboardCorners(img, PATTERN, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        print("sem tabuleiro:", fname)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None,
    flags=cv.CALIB_FIX_K3)

print("resolução (largura, altura):", gray.shape[::-1])
print("RMS (px):", ret)
print("K =\n", mtx)
print("dist =", dist.ravel())


# Erro de reprojeção médio (parte final do tutorial)
mean_error = 0
per_image = []
for i in range(len(objpoints)):
    proj, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    proj = proj.reshape(-1, 2)
    det = imgpoints[i].reshape(-1, 2)
    error = cv.norm(det, proj, cv.NORM_L2) / np.sqrt(len(proj))   # RMS real, em px
    per_image.append(error)
    print(f"imagem {i}: {error:.3f} px")
    mean_error += error
print("erro médio:", mean_error / len(objpoints))

print("resolução (largura, altura):", gray.shape[::-1])

# ADAPTAÇÃO 3: guardar para os outros módulos usarem
np.savez(BASE / "camera.npz", K=mtx, dist=dist, size=gray.shape[::-1])
print("guardado em:", BASE / "camera.npz")

