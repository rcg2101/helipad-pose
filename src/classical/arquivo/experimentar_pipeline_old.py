"""
Exploracao manual da deteccao classica do H, passo a passo.

Abrir este ficheiro no VS Code ou Spyder e executa-o diretamente.
Altera IMAGE_PATH para escolher outra fotografia.

No fim de cada operacao, o resultado abre na mesma janela. Carrega numa tecla
para avancar. A deteccao final usa Otsu para tinta escura e mostra os vertices
numerados. Nada e guardado a menos que guardar_imagens seja True.
"""

import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = ROOT / "data" / "dataset" / "images" / "val" / "synth00__00001.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "classical_out" / "experimento"
guardar_imagens = False
window_name = "Exploracao manual da pipeline classica"

if guardar_imagens:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


H_POINTS = np.array(
    [
        [-2, -2.5], [-1, -2.5], [-1, -0.5], [1, -0.5],
        [1, -2.5], [2, -2.5], [2, 2.5], [1, 2.5],
        [1, 0.5], [-1, 0.5], [-1, 2.5], [-2, 2.5],
    ],
    dtype=np.float32,
)


def _polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)) / 2)


def _project_points(homography, points):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = homogeneous @ homography.T
    return projected[:, :2] / projected[:, 2:3]


def _fit_contour(contour, mask):
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    model_area = _polygon_area(H_POINTS)
    best = None
    for epsilon in np.linspace(0.008, 0.05, 12) * perimeter:
        polygon = cv2.approxPolyDP(contour, float(epsilon), True).reshape(-1, 2)
        if len(polygon) != len(H_POINTS):
            continue
        polygon = polygon.astype(np.float64)
        if model_area * _polygon_area(polygon) < 0:
            polygon = polygon[::-1]

        for shift in range(len(polygon)):
            image_points = np.roll(polygon, shift, axis=0)
            homography, _ = cv2.findHomography(H_POINTS, image_points, 0)
            if homography is None:
                continue

            points = _project_points(homography, H_POINTS)
            if not np.isfinite(points).all():
                continue
            error = float(
                np.sqrt(np.mean(np.sum((points - image_points) ** 2, axis=1)))
            )

            top = points[[0, 1, 4, 5]].mean(axis=0)
            bottom = points[[6, 7, 10, 11]].mean(axis=0)
            direction = top - bottom
            up_angle = abs(math.degrees(math.atan2(direction[0], -direction[1])))
            if up_angle > 90:
                homography = homography @ np.diag([-1.0, -1.0, 1.0])
                points = _project_points(homography, H_POINTS)
                if not np.isfinite(points).all():
                    continue

            predicted_mask = np.zeros_like(mask)
            cv2.fillPoly(predicted_mask, [np.round(points).astype(np.int32)], 255)
            intersection = cv2.countNonZero(cv2.bitwise_and(predicted_mask, mask))
            union = cv2.countNonZero(cv2.bitwise_or(predicted_mask, mask))
            iou = intersection / max(union, 1)

            top = points[[0, 1, 4, 5]].mean(axis=0)
            bottom = points[[6, 7, 10, 11]].mean(axis=0)
            direction = top - bottom
            up_angle = abs(math.degrees(math.atan2(direction[0], -direction[1])))
            score = (error, up_angle)
            if best is None or score < best["score"]:
                best = {"points": points, "iou": iou, "score": score}
    return best


def _detect_h(contours, hierarchy, mask):
    if hierarchy is None:
        return None

    candidates = []
    for index, links in enumerate(hierarchy):
        contour = contours[index]
        area = cv2.contourArea(contour)
        if links[3] != -1 or area < 60 or len(contour) < 12:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0 or not 0.45 <= area / hull_area <= 0.78:
            continue

        h_mask = np.zeros_like(mask)
        cv2.drawContours(h_mask, [contour], -1, 255, cv2.FILLED)
        fit = _fit_contour(contour, h_mask)
        if fit is not None:
            candidates.append(fit)

    if not candidates:
        return None
    best = max(candidates, key=lambda fit: fit["iou"])
    return best if best["iou"] >= 0.75 else None


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


# Passo 5: remover pequenos pontos isolados da mascara de Otsu.
kernel = np.ones((3, 3), dtype=np.uint8)
clean_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)
cv2.imshow(window_name, clean_mask)
cv2.waitKey(0)
if guardar_imagens and not cv2.imwrite(
    str(OUTPUT_DIR / "05_mascara_limpa.png"), clean_mask
):
    raise OSError("Nao foi possivel guardar 05_mascara_limpa.png")


# Passo 6: obter os contornos da mascara e desenha-los sobre a fotografia.
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
    str(OUTPUT_DIR / "06_contornos.png"), contour_image
):
    raise OSError("Nao foi possivel guardar 06_contornos.png")


# Passo 7: escolher o melhor contorno, validar o H e numerar os vertices.
fit = _detect_h(
    contours,
    hierarchy[0] if hierarchy is not None else None,
    clean_mask,
)
detection_image = image.copy()
if fit is not None:
    points = fit["points"]
    cv2.polylines(
        detection_image,
        [np.round(points).astype(np.int32).reshape(-1, 1, 2)],
        True,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    for index, (x, y) in enumerate(points):
        center = (int(round(x)), int(round(y)))
        cv2.circle(detection_image, center, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            detection_image,
            str(index),
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    message = f"H detetado | IoU {fit['iou']:.2f}"
    color = (0, 255, 0)
else:
    message = "H nao detetado"
    color = (0, 165, 255)
cv2.putText(
    detection_image,
    message,
    (10, 28),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    color,
    2,
)
cv2.imshow(window_name, detection_image)
cv2.waitKey(0)
if guardar_imagens and not cv2.imwrite(
    str(OUTPUT_DIR / "07_deteccao_numerada.png"), detection_image
):
    raise OSError("Nao foi possivel guardar 07_deteccao_numerada.png")

cv2.destroyAllWindows()
print(f"H detetado: {'sim' if fit is not None else 'nao'}")
if guardar_imagens:
    print(f"Passos guardados em: {OUTPUT_DIR}")
