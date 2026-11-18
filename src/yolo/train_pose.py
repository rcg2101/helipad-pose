from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = ROOT / "data" / "dataset_v3" / "data.yaml" # path para o yaml
PROJECT = ROOT / "runs" / "train"                      # path para salvar os resultados do treino
MODEL_PATH = ROOT / "weights" / "pretrained" / "yolo26n-pose.pt" # pesos base
RUN_NAME = "y26n-7-db-2" # nome do run: modelo-num-basededados-num

def main() -> None:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Pretrained model not found: {MODEL_PATH}")

    run_dir = PROJECT / RUN_NAME
    if run_dir.exists():
        raise FileExistsError(
            f"Training output already exists: {run_dir}. "
            "Choose a new RUN_NAME."
        )

    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(DATA_YAML),
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=False,
        epochs=150,
        patience=30,
        #imgsz=1280, # estava 640
        #batch=4, # estava 16
        # DGX ------
        imgsz=640,
        batch=16,
        device=1,
        workers=8,
        amp=True,
        # DGX ------
        seed=0,
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
        max_det=1,
    )

    run_dir = Path(model.trainer.save_dir)
    print(f"Treino guardado em: {run_dir}")
    print(f"Melhores pesos: {run_dir / 'weights' / 'best.pt'}")
    print(f"Últimos pesos: {run_dir / 'weights' / 'last.pt'}")

    # Explicação da diferença entre best.pt e last.pt:
    # - best.pt: contém os pesos do modelo que obteve a melhor performance
    #   durante o treino, de acordo com a métrica de avaliação escolhida.
    # - last.pt: contém os pesos do modelo na última época de treino, independentemente
    #   de sua performance. Pode ser útil para continuar o treino ou para análise posterior.


if __name__ == "__main__":
    main()
