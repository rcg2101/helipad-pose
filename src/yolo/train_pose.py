from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

# Change this to the existing data.yaml for another dataset.
DATA_YAML = ROOT / "data" / "dataset" / "data.yaml"
MODEL_WEIGHTS = ROOT / "yolo11n-pose.pt"
PROJECT = ROOT / "runs" / "pose"
RUN_NAME = "h12"


if not DATA_YAML.is_file():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")
if not MODEL_WEIGHTS.is_file():
    raise FileNotFoundError(f"Starting model weights not found: {MODEL_WEIGHTS}")

model = YOLO(str(MODEL_WEIGHTS))
model.train(
    data=str(DATA_YAML),
    epochs=150,
    patience=30,
    imgsz=640,
    batch=16,
    seed=0,
    project=str(PROJECT),
    name=RUN_NAME,
    exist_ok=False,
    fliplr=0.5,
    flipud=0.0,
    degrees=0.0,
    scale=0.5,
    translate=0.1,
    perspective=0.0,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    mosaic=1.0,
    close_mosaic=10,
)

run_dir = Path(model.trainer.save_dir)
print(f"Treino guardado em: {run_dir}")
print(f"Melhores pesos: {run_dir / 'weights' / 'best.pt'}")
print(f"Últimos pesos: {run_dir / 'weights' / 'last.pt'}")
