#!/usr/bin/env python3
"""Deteção clássica do H e visualização das fases intermédias.

Como correr a partir da raiz do projeto:
    python src/classical/classical_h.py image "C:\\caminho\\para\\foto.jpg"
    python src/classical/classical_h.py image "C:\\caminho\\para\\foto.jpg" --save-dir classical_out 
    python src/classical/classical_h.py image data/dataset/images/val
    python src/classical/classical_h.py video --source 0
    python src/classical/classical_h.py video --source "C:\\caminho\\para\\video.mp4"

Os mosaicos das imagens são guardados por omissão em
    src/classical/classical_out/
Para escolher outra pasta, acrescentar `--save-dir "C:\\caminho\\para\\pasta"`.
Na janela: m/0 mostra o mosaico, 1-9 seleciona fases, [/ ] percorre todas as
fases e q fecha. A webcam/vídeo é apenas visualizada, não é gravada.

Pipeline:
    1. Converte cada frame para cinzento e aplica um desfoque ligeiro.
    2. Cria máscaras binárias com Otsu e limiarização adaptativa, para tinta
       escura ou clara.
    3. Procura contornos que possam formar o anel e valida forma, tamanho,
       proporção e concentricidade das elipses.
    4. Procura o H dentro do anel; se o anel estiver cortado ou não for
       detetado, tenta encontrar o H pelo contorno isolado.
    5. Ajusta uma homografia entre os 12 vértices do modelo do H e o contorno
       encontrado. A simetria de 180 graus é resolvida escolhendo a orientação
       em que o topo do H aponta mais para cima na imagem.
    6. Calcula a sobreposição (IoU) entre a máscara prevista e a segmentada;
       aceita a deteção se atingir o limiar configurado.
    7. Mostra o original, cinzento, máscaras, contornos, máscara do H e
       resultado com os 12 vértices numerados.
"""

import argparse
import ctypes
import math
import sys
from pathlib import Path

import cv2
import numpy as np


# The 12 outline vertices, in the same order used by the YOLO-pose dataset.
H_POINTS = np.array(
    [
        [-2, -2.5], [-1, -2.5], [-1, -0.5], [1, -0.5],
        [1, -2.5], [2, -2.5], [2, 2.5], [1, 2.5],
        [1, 0.5], [-1, 0.5], [-1, 2.5], [-2, 2.5],
    ],
    dtype=np.float32,
)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "classical_out"


def _odd(value):
    value = int(round(value))
    return value if value % 2 else value + 1


def threshold_masks(gray, polarity):
    """Create Otsu and adaptive masks; foreground (ink) is white."""
    height, width = gray.shape
    max_block = min(height, width)
    if max_block % 2 == 0:
        max_block -= 1
    if max_block < 3:
        raise ValueError("A imagem tem de ter pelo menos 3 px em cada dimensão.")

    block_sizes = {
        "Adaptativa fina": min(max_block, max(15, _odd(max_block / 24))),
        "Adaptativa larga": min(max_block, max(31, _odd(max_block / 6))),
    }
    if block_sizes["Adaptativa fina"] == block_sizes["Adaptativa larga"]:
        block_sizes["Adaptativa larga"] = max_block

    polarities = ("dark", "light") if polarity == "auto" else (polarity,)
    for mode in polarities:
        invert = mode == "dark"
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        suffix = "tinta escura" if invert else "tinta clara"

        _, otsu = cv2.threshold(
            gray, 0, 255, threshold_type | cv2.THRESH_OTSU
        )
        yield f"Otsu ({suffix})", otsu

        adaptive_type = (
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        )
        for name, block_size in block_sizes.items():
            mask = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                adaptive_type,
                block_size,
                7 if name == "Adaptativa larga" else 5,
            )
            yield f"{name} ({suffix})", mask


def _ellipse_residual(contour, ellipse):
    (cx, cy), (width, height), angle = ellipse
    a, b = max(width, 1e-6) / 2, max(height, 1e-6) / 2
    angle = math.radians(angle)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    points = contour.reshape(-1, 2).astype(np.float64) - (cx, cy)
    x = points[:, 0] * cos_a + points[:, 1] * sin_a
    y = -points[:, 0] * sin_a + points[:, 1] * cos_a
    radius = np.sqrt((x / a) ** 2 + (y / b) ** 2)
    return float(np.mean(np.abs(radius - 1)))


def _find_rings(contours, hierarchy, min_ring_px):
    if hierarchy is None:
        return []

    rings = []
    for outer_index, links in enumerate(hierarchy):
        first_child = int(links[2])
        outer = contours[outer_index]
        if links[3] != -1 or first_child == -1 or len(outer) < 20:
            continue

        holes = []
        child_index = first_child
        while child_index != -1:
            holes.append(child_index)
            child_index = int(hierarchy[child_index][0])
        if not holes:
            continue

        inner_index = max(holes, key=lambda i: cv2.contourArea(contours[i]))
        inner = contours[inner_index]
        if len(inner) < 20:
            continue

        outer_ellipse = cv2.fitEllipse(outer)
        inner_ellipse = cv2.fitEllipse(inner)
        outer_axes = np.asarray(outer_ellipse[1], dtype=float)
        inner_axes = np.asarray(inner_ellipse[1], dtype=float)
        outer_area = math.pi * np.prod(outer_axes / 2)
        inner_area = math.pi * np.prod(inner_axes / 2)
        if min(outer_axes) < min_ring_px or outer_area <= 0:
            continue
        if min(outer_axes) / max(outer_axes) < 0.25:
            continue

        ratio = inner_area / outer_area
        if not 0.50 <= ratio <= 0.78:
            continue
        residual_limit = 0.04 + 0.7 / max(min(inner_axes) / 2, 1)
        if max(
            _ellipse_residual(outer, outer_ellipse),
            _ellipse_residual(inner, inner_ellipse),
        ) > residual_limit:
            continue

        outer_center = np.asarray(outer_ellipse[0])
        inner_center = np.asarray(inner_ellipse[0])
        if np.linalg.norm(outer_center - inner_center) / max(max(outer_axes) / 2, 1) > 0.12:
            continue

        rings.append(
            {
                "outer": outer_index,
                "inner": inner_index,
                "ellipse": inner_ellipse,
                "area": inner_area,
            }
        )
    return rings


def _inside_ellipse(point, ellipse):
    (cx, cy), (width, height), angle = ellipse
    angle = math.radians(angle)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx, dy = point[0] - cx, point[1] - cy
    x = (dx * cos_a + dy * sin_a) / max(width / 2, 1e-6)
    y = (-dx * sin_a + dy * cos_a) / max(height / 2, 1e-6)
    return x * x + y * y


def _find_h_contour(contours, hierarchy, ring, min_h_area):
    (cx, cy), (width, height), _ = ring["ellipse"]
    inner_area = math.pi * width * height / 4
    candidates = []

    for index, links in enumerate(hierarchy):
        contour = contours[index]
        area = cv2.contourArea(contour)
        if links[3] != -1 or len(contour) < 12:
            continue
        if not max(min_h_area, 0.08 * inner_area) <= area <= 0.45 * inner_area:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        center = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )
        if _inside_ellipse(center, ring["ellipse"]) > 0.25:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0 or not 0.45 <= area / hull_area <= 0.78:
            continue
        candidates.append((area, index))

    return max(candidates)[1] if candidates else None


def _find_h_without_ring(contours, hierarchy, mask, min_h_area):
    candidates = []
    for index, links in enumerate(hierarchy):
        contour = contours[index]
        area = cv2.contourArea(contour)
        if links[3] != -1 or area < min_h_area or len(contour) < 12:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0 or not 0.45 <= area / hull_area <= 0.78:
            continue

        h_mask = np.zeros_like(mask)
        cv2.drawContours(h_mask, [contour], -1, 255, cv2.FILLED)
        fit = _fit_homography(contour, h_mask)
        if fit is not None:
            candidates.append((fit["iou"], index, h_mask, fit))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def _apply_homography(homography, points):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = homogeneous @ homography.T
    return projected[:, :2] / projected[:, 2:3]


def _polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)) / 2)


def _fit_homography(contour, mask):
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    model_area = _polygon_area(H_POINTS)
    height, width = mask.shape
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
            if homography is None or not np.isfinite(homography).all():
                continue
            projected = _apply_homography(homography, H_POINTS)
            error = float(
                np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1)))
            )
            if best is not None and error > best["error"] + 0.25:
                continue

            top = projected[[0, 1, 4, 5]].mean(axis=0)
            bottom = projected[[6, 7, 10, 11]].mean(axis=0)
            direction = top - bottom
            up_angle = abs(math.degrees(math.atan2(direction[0], -direction[1])))
            if up_angle > 90:
                homography = homography @ np.diag([-1.0, -1.0, 1.0])
                projected = _apply_homography(homography, H_POINTS)

            predicted_mask = np.zeros_like(mask)
            cv2.fillPoly(
                predicted_mask,
                [np.round(projected).astype(np.int32)],
                255,
            )
            intersection = cv2.countNonZero(
                cv2.bitwise_and(predicted_mask, mask)
            )
            union = cv2.countNonZero(cv2.bitwise_or(predicted_mask, mask))
            iou = intersection / max(union, 1)
            top = projected[[0, 1, 4, 5]].mean(axis=0)
            bottom = projected[[6, 7, 10, 11]].mean(axis=0)
            direction = top - bottom
            up_angle = abs(math.degrees(math.atan2(direction[0], -direction[1])))

            score = (error, up_angle)
            if best is None or score < best["score"]:
                best = {
                    "homography": homography,
                    "points": projected,
                    "error": error,
                    "iou": iou,
                    "score": score,
                }

    if best is None:
        return None
    best["iou"] = float(best["iou"])
    best["error"] = float(best["error"])
    if not (0 <= best["iou"] <= 1) or not np.isfinite(best["points"]).all():
        return None
    if np.any(best["points"][:, 0] < -width) or np.any(best["points"][:, 0] > 2 * width):
        return None
    if np.any(best["points"][:, 1] < -height) or np.any(best["points"][:, 1] > 2 * height):
        return None
    return best


def _analyse_mask(mask, min_ring_px, min_h_area):
    contours, raw_hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    hierarchy = raw_hierarchy[0] if raw_hierarchy is not None else None
    result = {
        "contours": contours,
        "hierarchy": hierarchy,
        "rings": [],
        "h_contour": None,
        "h_mask": np.zeros_like(mask),
        "fit": None,
    }
    if hierarchy is None:
        return result

    result["rings"] = _find_rings(contours, hierarchy, min_ring_px)
    candidates = []
    for ring in result["rings"]:
        h_index = _find_h_contour(contours, hierarchy, ring, min_h_area)
        if h_index is None:
            continue
        h_contour = contours[h_index]
        h_mask = np.zeros_like(mask)
        cv2.drawContours(h_mask, [h_contour], -1, 255, cv2.FILLED)
        fit = _fit_homography(h_contour, h_mask)
        if fit is not None:
            candidates.append((fit["iou"], h_index, h_mask, fit))

    if not candidates:
        fallback = _find_h_without_ring(
            contours, hierarchy, mask, min_h_area
        )
        if fallback is not None:
            candidates.append(fallback)

    if candidates:
        _, h_index, h_mask, fit = max(candidates, key=lambda item: item[0])
        result.update(h_contour=contours[h_index], h_mask=h_mask, fit=fit)
    return result


def _draw_keypoints(image, points):
    for index, (x, y) in enumerate(points):
        center = (int(round(x)), int(round(y)))
        cv2.circle(image, center, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(index),
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )


def detect(frame, polarity="auto", blur=1.0, min_ring_px=24, min_h_area=60, min_iou=0.75):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur)

    masks = list(threshold_masks(gray, polarity))
    analyses = [
        (name, mask, _analyse_mask(mask, min_ring_px, min_h_area))
        for name, mask in masks
    ]
    best = max(
        analyses,
        key=lambda item: item[2]["fit"]["iou"] if item[2]["fit"] else 0,
    )
    best_name, best_mask, details = best
    fit = details["fit"]
    found = fit is not None and fit["iou"] >= min_iou

    contour_view = frame.copy()
    for index, contour in enumerate(details["contours"]):
        color = (255, 140, 0)
        if details["hierarchy"] is not None and details["hierarchy"][index][3] != -1:
            color = (0, 200, 255)
        cv2.drawContours(contour_view, [contour], -1, color, 1)

    for ring in details["rings"]:
        cv2.ellipse(
            contour_view,
            cv2.fitEllipse(details["contours"][ring["outer"]]),
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    if details["h_contour"] is not None:
        cv2.drawContours(contour_view, [details["h_contour"]], -1, (0, 0, 255), 2)

    result = frame.copy()
    if found:
        cv2.polylines(
            result,
            [np.round(fit["points"]).astype(np.int32).reshape(-1, 1, 2)],
            True,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        _draw_keypoints(result, fit["points"])
        message = f"H detetado | IoU {fit['iou']:.2f} | mascara: {best_name}"
        color = (0, 255, 0)
    else:
        score = fit["iou"] if fit is not None else 0
        message = f"H nao detetado | IoU {score:.2f} | mascara: {best_name}"
        color = (0, 165, 255)
    cv2.putText(result, message, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    views = [
        ("Original", frame),
        ("Cinza + blur", gray),
    ]
    views.extend((name, mask) for name, mask, _ in analyses)
    views.extend(
        [
            (f"Contornos ({best_name})", contour_view),
            ("Mascara do H", details["h_mask"]),
            ("Resultado", result),
        ]
    )
    return {
        "found": found,
        "iou": fit["iou"] if fit is not None else 0.0,
        "points": fit["points"] if found else None,
        "views": views,
    }


def _to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()


def _screen_size():
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return 1280, 720


def _make_mosaic(views, tile_width=400, columns=3):
    screen_width, screen_height = _screen_size()
    rows = math.ceil(len(views) / columns)
    tile_width = min(
        tile_width,
        max(1, (screen_width - 40) // columns),
        max(1, int((screen_height - 100) / (rows * 0.62))),
    )
    tile_height = int(tile_width * 0.62)
    tiles = []
    for index, (name, image) in enumerate(views, 1):
        image = _to_bgr(image)
        height, width = image.shape[:2]
        scale = min(tile_width / width, tile_height / height)
        resized = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        tile = np.full((tile_height, tile_width, 3), 28, dtype=np.uint8)
        y = (tile_height - resized.shape[0]) // 2
        x = (tile_width - resized.shape[1]) // 2
        tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        title = f"Passo {index} - {name}"
        cv2.putText(tile, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        tiles.append(tile)

    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 28))
    rows = [
        np.hstack(tiles[index:index + columns])
        for index in range(0, len(tiles), columns)
    ]
    return np.vstack(rows)


def _display(views, selected):
    if selected == 0:
        return _make_mosaic(views)
    index = min(selected - 1, len(views) - 1)
    image = _to_bgr(views[index][1])
    cv2.putText(
        image,
        f"Passo {index + 1} - {views[index][0]} (m = mosaico)",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    return image


def _image_files(paths):
    files = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(
                sorted(
                    file for file in path.iterdir()
                    if file.suffix.lower() in IMAGE_EXTENSIONS
                )
            )
        else:
            files.append(path)
    return files


def _process_options(args):
    return {
        "polarity": args.polarity,
        "blur": args.blur,
        "min_ring_px": args.min_ring_px,
        "min_h_area": args.min_h_area,
        "min_iou": args.min_iou,
    }


def show_images(args):
    files = _image_files(args.paths)
    if not files:
        raise FileNotFoundError("Não foram encontradas imagens nos caminhos indicados.")

    save_dir = Path(args.save_dir)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Não foi possível ler: {path}")
            continue
        detection = detect(frame, **_process_options(args))
        print(f"{path.name}: {'H detetado' if detection['found'] else 'H não detetado'}")
        if save_dir:
            cv2.imwrite(
                str(save_dir / f"{path.stem}_mascaras.png"),
                _make_mosaic(detection["views"]),
            )

        selected = 0
        while True:
            cv2.imshow("Detecao classica - imagem", _display(detection["views"], selected))
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if key in (ord("m"), ord("0")):
                selected = 0
                continue
            if ord("1") <= key <= ord("9"):
                selected = min(key - ord("0"), len(detection["views"]))
                continue
            if key in (ord("]"), ord("n")):
                selected = (selected + 1) % (len(detection["views"]) + 1)
                continue
            if key in (ord("["), ord("b")):
                selected = (selected - 1) % (len(detection["views"]) + 1)
                continue
            break
    cv2.destroyAllWindows()


def _parse_source(source):
    return int(source) if source.isdecimal() else source


def show_video(args):
    capture = cv2.VideoCapture(_parse_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {args.source}")

    selected = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detection = detect(frame, **_process_options(args))
            cv2.imshow("Detecao classica - video", _display(detection["views"], selected))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("m"), ord("0")):
                selected = 0
            elif ord("1") <= key <= ord("9"):
                selected = min(key - ord("0"), len(detection["views"]))
            elif key in (ord("]"), ord("n")):
                selected = (selected + 1) % (len(detection["views"]) + 1)
            elif key in (ord("["), ord("b")):
                selected = (selected - 1) % (len(detection["views"]) + 1)
    finally:
        capture.release()
        cv2.destroyAllWindows()


def add_detection_options(parser):
    parser.add_argument("--polarity", choices=("dark", "light", "auto"), default="auto")
    parser.add_argument("--blur", type=float, default=1.0)
    parser.add_argument("--min-ring-px", type=float, default=24)
    parser.add_argument("--min-h-area", type=float, default=60)
    parser.add_argument("--min-iou", type=float, default=0.80)


def main():
    parser = argparse.ArgumentParser(
        description="Detecao classica do H e visualizacao das mascaras."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="Mostrar imagem(ns) e mascaras")
    image_parser.add_argument("paths", nargs="+", help="Imagem ou pasta de imagens")
    image_parser.add_argument(
        "--save-dir",
        default=str(DEFAULT_OUTPUT),
        help=f"Pasta para os mosaicos (default: {DEFAULT_OUTPUT})",
    )
    add_detection_options(image_parser)
    image_parser.set_defaults(func=show_images)

    video_parser = subparsers.add_parser("video", help="Mostrar webcam ou ficheiro de video")
    video_parser.add_argument("--source", default="0", help="Indice da webcam ou caminho do video")
    add_detection_options(video_parser)
    video_parser.set_defaults(func=show_video)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
