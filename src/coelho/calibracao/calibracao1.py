import glob
from pathlib import Path
 
import numpy as np
import cv2 as cv
 
# ---------------------------------------------------------------------------
# Parâmetros do tabuleiro (ADAPTAR se mudares de tabuleiro)
# ---------------------------------------------------------------------------
# Cantos INTERNOS (colunas, linhas), ou seja, os pontos onde 4 quadrados se
# encontram, sem contar a borda exterior. O meu tabuleiro tem 7x9 quadrados,
# logo 6x8 cantos internos.
PATTERN = (6, 8)
# Lado real de um quadrado impresso, em metros (2,5 cm). Medido com régua.
# Não influencia o K (que está em píxeis); só dá escala métrica aos tvecs.
SQUARE = 0.025
 
# Mostrar cada foto com os cantos desenhados (útil para verificar, mas pára 0,5 s por foto)
SHOW = True
 
# Critério de paragem do refinamento sub-pixel dos cantos:
# para ao fim de 30 iterações ou quando a correção for menor que 0,001 px.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# ---------------------------------------------------------------------------
# Pontos 3D do tabuleiro (no referencial do próprio tabuleiro)
# ---------------------------------------------------------------------------
# O tabuleiro é plano, por isso Z = 0. Os cantos formam uma grelha:
# (0,0,0), (1,0,0), ..., (5,0,0), (0,1,0), ..., (5,7,0), em "unidades de quadrado".
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
# ADAPTAÇÃO 1: multiplicar pelo lado do quadrado para ficar em metros.
objp *= SQUARE
 
objpoints = []   # pontos 3D (iguais em todas as fotos, porque o tabuleiro é o mesmo)
imgpoints = []   # pontos 2D detetados em cada foto (em píxeis)
 
# ---------------------------------------------------------------------------
# Localizar as fotos
# ---------------------------------------------------------------------------
# ADAPTAÇÃO 2: o caminho é construído a partir da localização do script, e não
# da "pasta de trabalho atual" (que muda conforme a forma como o script é
# corrido e foi a causa do erro "gray is not defined" no início).
BASE = Path(__file__).resolve().parent                  
CALIB_DIR = BASE.parent.parent / "data" / "calib"       
 
images = []
for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):       # fotos de telemóvel podem ter .JPG
    images += glob.glob(str(CALIB_DIR / ext))
 
print("pasta de calibração:", CALIB_DIR)
print("imagens encontradas:", len(images))
 
# Falhar cedo e com mensagem clara se a pasta estiver vazia. Sem isto, o erro
# só aparece mais abaixo, como um NameError confuso.
if not images:
    raise FileNotFoundError(f"Nenhuma imagem encontrada em {CALIB_DIR}")
 
# ---------------------------------------------------------------------------
# Detetar os cantos em cada foto
# ---------------------------------------------------------------------------
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Procura a grelha completa de cantos internos. Só devolve ret=True se vir
    # TODOS os cantos (por isso o tabuleiro tem de aparecer inteiro na foto).
    ret, corners = cv.findChessboardCorners(gray, PATTERN, None)
 
    if ret:
        objpoints.append(objp)
        # Refina cada canto para uma fração de píxel (janela de 11x11 em volta).
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
 
        if SHOW:
            # Desenha os cantos para confirmar visualmente que a deteção está certa
            cv.drawChessboardCorners(img, PATTERN, corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(500)
    else:
        print("sem tabuleiro:", fname)    # foto rejeitada (canto cortado, desfocada, PATTERN errado...)
cv.destroyAllWindows()
 
# ---------------------------------------------------------------------------
# Calibração
# ---------------------------------------------------------------------------
# Encontra o K e a distorção que melhor explicam todas as correspondências
# 3D <-> 2D de todas as fotos. gray.shape[::-1] dá (largura, altura).
#
# flags=CALIB_FIX_K3: fixa k3 = 0. Com apenas 17 fotos, o k3 saía muito grande
# (4,14) e a absorver ruído. Fixá-lo piorou o RMS só 0,017 px (0,458 -> 0,475),
# quase não mexeu em K e deu k1 e k2 mais plausíveis.
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None,
    flags=cv.CALIB_FIX_K3)
 
print("resolução (largura, altura):", gray.shape[::-1])
print("RMS (px):", ret)       # erro de reprojeção global; objetivo: < ~0,5 px
print("K =\n", mtx)
print("dist =", dist.ravel())  # [k1, k2, p1, p2, k3]
 
# ---------------------------------------------------------------------------
# Erro de reprojeção por imagem
# ---------------------------------------------------------------------------
# Projeta os pontos 3D com os parâmetros calibrados e compara com os cantos
# detetados. Serve para identificar fotos más (valores muito acima das outras).
# Divide-se por sqrt(N) para obter um RMS verdadeiro em píxeis, comparável com
# o "RMS (px)" acima. (O tutorial original divide por N e não é comparável.)
# O reshape(-1, 2) iguala os formatos (N,2): com o OpenCV 5 o cv.norm do
# tutorial dava erro de tipos.
mean_error = 0
per_image = []
for i in range(len(objpoints)):
    proj, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    proj = proj.reshape(-1, 2)
    det = imgpoints[i].reshape(-1, 2)
    error = cv.norm(det, proj, cv.NORM_L2) / np.sqrt(len(proj))
    per_image.append(error)
    print(f"imagem {i}: {error:.3f} px")
    mean_error += error
print("erro médio:", mean_error / len(objpoints))
 
# ---------------------------------------------------------------------------
# Guardar para os outros módulos
# ---------------------------------------------------------------------------
# ADAPTAÇÃO 3: ficheiro ao lado do script, independente da pasta de trabalho.
# "size" guarda a resolução de calibração para os outros scripts poderem
# verificar que a imagem tem o tamanho certo antes de usar o K.
np.savez(BASE / "camera.npz", K=mtx, dist=dist, size=gray.shape[::-1])
print("guardado em:", BASE / "camera.npz")