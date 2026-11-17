#!/usr/bin/env python3
"""Deteção Otsu do H na webcam. Executar sem argumentos."""

import math
import time

import cv2
import numpy as np

WEBCAM_INDEX = 0
MIN_RING_PX = 24
MIN_H_AREA = 60
MIN_IOU = 0.80


# The 12 outline vertices, in the same order used by the YOLO-pose dataset.
H_POINTS = np.array(
    [
        [-2, -2.5], [-1, -2.5], [-1, -0.5], [1, -0.5],
        [1, -2.5], [2, -2.5], [2, 2.5], [1, 2.5],
        [1, 0.5], [-1, 0.5], [-1, 2.5], [-2, 2.5],
    ],
    dtype=np.float32,
)


def otsu_mask(gray):
    _, otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return otsu


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


def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    details = _analyse_mask(otsu_mask(blurred), MIN_RING_PX, MIN_H_AREA)
    fit = details["fit"]
    found = fit is not None and fit["iou"] >= MIN_IOU
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
        message = f"H detetado | IoU {fit['iou']:.2f}"
        color = (0, 255, 0)
    else:
        score = fit["iou"] if fit is not None else 0
        message = f"H nao detetado | IoU {score:.2f}"
        color = (0, 165, 255)
    cv2.putText(result, message, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return result


def main():
    capture = cv2.VideoCapture(WEBCAM_INDEX)
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir a webcam {WEBCAM_INDEX}.")

    last_frame_time = None
    fps = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            display = detect(frame)
            fps_text = "FPS: --" if fps is None else f"FPS: {fps:.1f}"
            cv2.putText(
                display,
                fps_text,
                (10, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Detecao classica - video", display)
            key = cv2.waitKey(1) & 0xFF
            now = time.perf_counter()
            if last_frame_time is not None:
                elapsed = now - last_frame_time
                if elapsed > 0:
                    measured_fps = 1 / elapsed
                    fps = measured_fps if fps is None else 0.85 * fps + 0.15 * measured_fps
            last_frame_time = now
            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
