#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate v4 synthetic images with their generation settings overlaid.

Run directly, without arguments. Output goes to blur_lab_dataset/ and is meant
for visual inspection, not model training, because the text is drawn on images.
"""

import json
import sys
from pathlib import Path

import cv2

import gen_synthetic_v4 as generator


IMAGE_COUNT = 200
OUTPUT_DIR = Path(__file__).resolve().parent / "blur_lab_dataset"


def _settings_lines(settings):
    lines = []
    pose = settings.get("pose")
    if pose:
        attitude = pose["attitude_deg"]
        prefix = "H" if settings["has_h"] else "NEGATIVE | false target"
        lines.append(
            f"{prefix} | roll={attitude['roll']:+.1f} pitch={attitude['pitch']:+.1f} "
            f"yaw={attitude['yaw']:+.1f} deg | d={pose['distance_m']:.2f}m | "
            f"tilt={pose['tilt_deg']:.1f} deg"
        )
        lines.append(
            f"Pad={pose['pad_shape']} {pose['pad_size_units'][0]:.1f}x"
            f"{pose['pad_size_units'][1]:.1f}x | center={pose['center_px']} px | "
            f"partial requested={pose['partial_requested']} | inside={pose['keypoints_inside']} | "
            f"visible={pose['keypoints_visible']} | zeroed by blur={pose['keypoints_zeroed_by_blur']}"
        )
        lines.append(f"Pad geometry={pose['pad_geometry']}")
    else:
        lines.append("NEGATIVE | no H target")

    target = settings.get("target")
    if target:
        lines.append(f"Pad colors: paper BGR={target['paper_bgr']} | ink BGR={target['ink_bgr']}")
        if "h_scale" in target:
            lines.append(f"False H scale={target['h_scale']}")

    background = settings["background"]
    if background["type"] == "photo":
        lines.append(
            f"Background=photo:{background['source']} | crop={background['crop_xywh']} px | "
            f"mirrored={background['mirrored']}"
        )
    else:
        lines.append(f"Background={background['type']}")
    confusers = settings["confusers"]
    if confusers:
        for index, confuser in enumerate(confusers, 1):
            lines.append(f"Confuser {index}: {confuser}")
    else:
        lines.append("Confusers=none")

    blur = settings["blur"]
    lines.append(
        f"Motion blur={blur['length_x']:.2f}x/{blur['length_px']:.1f}px | "
        f"direction={blur['angle_deg']} deg | scope={blur['scope']}"
    )
    occlusion = settings["occlusion"]
    if occlusion:
        lines.append(
            f"Occlusion={occlusion['shape']} center={occlusion['center_px']} px | "
            f"size={occlusion['size_px']} px | angle={occlusion['angle_deg']} deg | "
            f"color BGR={occlusion['color_bgr']}"
        )
    else:
        lines.append("Occlusion=none")

    light = settings["photometric"]
    shadow = light["shadow"]
    shadow_text = "none" if shadow is None else (
        f"opacity={shadow['opacity']} blur={shadow['blur_px']}px vertices={shadow['vertices_px']}"
    )
    lines.append(
        f"Light: gain={light['gain']} cast BGR={light['color_cast']} | "
        f"gradient={light['gradient']} at {light['gradient_angle_deg']} deg | "
        f"shadow={shadow_text}"
    )
    lines.append(
        f"Gamma={light['gamma']} | lens blur sigma={light['lens_blur_sigma_px']}px | "
        f"noise sigma={light['noise_sigma']} | JPEG quality={settings['jpeg_quality']}"
    )
    return lines


def _draw_settings(image, lines):
    line_height = 19
    panel_height = 10 + line_height * len(lines)
    cv2.rectangle(image, (0, 0), (image.shape[1], panel_height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (8, 17 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def annotate_images():
    image_dir = OUTPUT_DIR / "images" / generator.SPLIT
    settings_dir = OUTPUT_DIR / "settings" / generator.SPLIT
    label_dir = OUTPUT_DIR / "labels" / generator.SPLIT
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No generated images found in {image_dir}")

    for index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise OSError(f"Could not read generated image: {image_path}")

        settings_path = settings_dir / f"{image_path.stem}.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        image = _draw_settings(image, _settings_lines(settings))
        if not cv2.imwrite(str(image_path), image):
            raise OSError(f"Could not write settings overlay to: {image_path}")

        if not (label_dir / f"{image_path.stem}.txt").exists():
            raise FileNotFoundError(f"Missing generated label for {image_path.stem}")
        if index % 10 == 0 or index == len(images):
            print(f"Annotated settings: {index}/{len(images)} images")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    generator.N_IMAGES = IMAGE_COUNT
    generator.OUT_DIR = OUTPUT_DIR
    generator.generate()
    generator.verify()
    annotate_images()
    print(f"Visual-inspection dataset with settings overlays: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
