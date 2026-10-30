"""
Exploracao passo a passo da segmentacao classica do H.

Abre este ficheiro no VS Code ou Spyder e executa-o diretamente.
Altera IMAGE_PATH para experimentar outra fotografia. Nao precisa de
argumentos nem de comandos no terminal.

O script guarda cada resultado em src/classical/classical_out/experimento/
e mostra-os, um de cada vez, na mesma janela. Carrega numa tecla para avancar
ou em q para fechar a janela.
"""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = ROOT / "data" / "dataset" / "images" / "val" / "synth00__00001.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "classical_out" / "experimento"


# 1. Carregar a imagem escolhida.
image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(f"Nao foi possivel abrir a imagem: {IMAGE_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Converter para tons de cinzento: simplifica a comparacao de intensidade.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 3. Reduzir ruido antes de criar a mascara.
blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)


# 4. Criar uma mascara com Otsu, assumindo tinta escura num fundo claro.
_, otsu_dark = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
)


# 5. Criar uma mascara adaptativa, util quando a iluminacao varia na imagem.
adaptive_dark = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    51,
    7,
)


# 6. Remover pequenos pontos isolados com abertura morfologica.
kernel = np.ones((3, 3), dtype=np.uint8)
otsu_clean = cv2.morphologyEx(otsu_dark, cv2.MORPH_OPEN, kernel)


# 7. Encontrar e desenhar os contornos da mascara de Otsu limpa.
contours, hierarchy = cv2.findContours(
    otsu_clean.copy(),
    cv2.RETR_CCOMP,
    cv2.CHAIN_APPROX_SIMPLE,
)
contour_view = image.copy()
cv2.drawContours(contour_view, contours, -1, (0, 200, 255), 1)


# Guarda todas as fases para comparar, mesmo que se feche a janela antes do fim.
steps = [
    ("01_original.png", "1 - Original", image),
    ("02_cinzento.png", "2 - Tons de cinzento", gray),
    ("03_desfoque.png", "3 - Desfoque gaussiano", blurred),
    ("04_otsu.png", "4 - Mascara Otsu (tinta escura)", otsu_dark),
    ("05_adaptativa.png", "5 - Mascara adaptativa", adaptive_dark),
    ("06_otsu_limpa.png", "6 - Otsu apos abertura morfologica", otsu_clean),
    ("07_contornos.png", "7 - Contornos da mascara Otsu", contour_view),
]

for filename, title, result in steps:
    output_path = OUTPUT_DIR / filename
    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"Nao foi possivel guardar o resultado: {output_path}")

print(f"Imagem: {IMAGE_PATH}")
print(f"Mascaras e resultados guardados em: {OUTPUT_DIR}")
print(f"Contornos encontrados: {len(contours)}")
print("Na janela: qualquer tecla avanca; q fecha.")

window_name = "Exploracao da pipeline classica"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1000, 650)

for _, title, result in steps:
    display = result
    if result.ndim == 2:
        display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        display,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.imshow(window_name, display)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
