"""
Exploracao manual da deteccao classica do H, passo a passo.

Abre este ficheiro no VS Code ou Spyder e executa-o diretamente.
Altera IMAGE_PATH para escolher outra fotografia.

No fim de cada operacao, o resultado abre na mesma janela. Carrega numa tecla
para avancar. A deteccao final e feita pela pipeline classica e mostra os
vertices numerados. Nada e guardado a menos que guardar_imagens seja True.
"""

from pathlib import Path

import cv2
import numpy as np

from classical_h import detect


ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = ROOT / "data" / "dataset" / "images" / "val" / "synth00__00001.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "classical_out" / "experimento"
guardar_imagens = False
window_name = "Exploracao manual da pipeline classica"

if guardar_imagens:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Passo 1: carregar e visualizar a fotografia original.
image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(f"Nao foi possivel abrir a imagem: {IMAGE_PATH}")
cv2.imshow(window_name, image)
cv2.waitKey(0)
if guardar_imagens and not cv2.imwrite(str(OUTPUT_DIR / "01_original.png"), image):
    raise OSError("Nao foi possivel guardar 01_original.png")


# Passo 2: converter para cinzento para trabalhar apenas com intensidade.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow(window_name, gray)
cv2.waitKey(0)
if guardar_imagens and not cv2.imwrite(str(OUTPUT_DIR / "02_cinzento.png"), gray):
    raise OSError("Nao foi possivel guardar 02_cinzento.png")


# Passo 3: aplicar um desfoque ligeiro para reduzir ruido.
blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
cv2.imshow(window_name, blurred)
cv2.waitKey(0)
if guardar_imagens and not cv2.imwrite(str(OUTPUT_DIR / "03_desfoque.png"), blurred):
    raise OSError("Nao foi possivel guardar 03_desfoque.png")


# Passo 4: criar uma mascara de tinta escura com Otsu.
_, otsu_mask = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
)
cv2.imshow(window_name, otsu_mask)
cv2.waitKey(0)
if guardar_imagens and not cv2.imwrite(str(OUTPUT_DIR / "04_otsu.png"), otsu_mask):
    raise OSError("Nao foi possivel guardar 04_otsu.png")


# Passo 5: criar uma mascara adaptativa para lidar com iluminacao desigual.
adaptive_mask = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    51,
    7,
)
cv2.imshow(window_name, adaptive_mask)
cv2.waitKey(0)
if guardar_imagens and not cv2.imwrite(
    str(OUTPUT_DIR / "05_adaptativa.png"), adaptive_mask
):
    raise OSError("Nao foi possivel guardar 05_adaptativa.png")


# Passo 6: remover pequenos pontos isolados da mascara de Otsu.
kernel = np.ones((3, 3), dtype=np.uint8)
clean_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)
cv2.imshow(window_name, clean_mask)
cv2.waitKey(0)
if guardar_imagens and not cv2.imwrite(
    str(OUTPUT_DIR / "06_mascara_limpa.png"), clean_mask
):
    raise OSError("Nao foi possivel guardar 06_mascara_limpa.png")


# Passo 7: obter os contornos da mascara e desenha-los sobre a fotografia.
contours, hierarchy = cv2.findContours(
    clean_mask.copy(),
    cv2.RETR_CCOMP,
    cv2.CHAIN_APPROX_NONE,
)
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 200, 255), 1)
cv2.imshow(window_name, contour_image)
cv2.waitKey(0)
if guardar_imagens and not cv2.imwrite(
    str(OUTPUT_DIR / "07_contornos.png"), contour_image
):
    raise OSError("Nao foi possivel guardar 07_contornos.png")


# Passo 8: executar a deteccao completa, validar o H e numerar os vertices.
# A funcao reutilizada evita duplicar a correspondencia geometrica da pipeline.
detection = detect(image)
detection_image = detection["views"][-1][1].copy()
cv2.imshow(window_name, detection_image)
cv2.waitKey(0)
if guardar_imagens and not cv2.imwrite(
    str(OUTPUT_DIR / "08_deteccao_numerada.png"), detection_image
):
    raise OSError("Nao foi possivel guardar 08_deteccao_numerada.png")

cv2.destroyAllWindows()
print(f"H detetado: {'sim' if detection['found'] else 'nao'}")
if guardar_imagens:
    print(f"Passos guardados em: {OUTPUT_DIR}")
