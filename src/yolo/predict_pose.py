from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
modelo = "y26-1-db-1"
MODEL_PATH = ROOT / "runs" / "train" / modelo / "weights" / "best.pt" # selecionar o modelo que quero

# ---------------------------------------Preencher as variáveis abaixo---------------------------------------
# 3 opções de SOURCE: webcam, vídeo ou imagem. Descomentar a desejada.

#SOURCE = 0 # para usar webcam
# SOURCE = ROOT / "data" / "dataset" / "videos" / "video.mp4" # para usar um vídeo
SOURCE = ROOT / "data" / "teste_beta" / "falhas_frames_vídeo_y21-1-1_1024_025_1" /"3.png" # para usar uma imagem



IMAGE_SIZE = 640 # 640x640, 512x512, 1024x1024, etc. (must be multiple of 32)
CONFIDENCE = 0.25 # 0.25 padrao. probabilidade mínima para considerar uma detecção válida. corresponde ao numero ao lado da bonding box
MAX_DETECTIONS = 300 # 300 padrao. número máximo de detecções a serem retornadas. 
                     # Manter valor elevado se quiser saber os falsos positivos, durante teste
                     # Usar 1 quando aplicar o modelo na prática
SHOW = True # mostrar a imagem com as detecções
SAVE = True # salvar a imagem com as detecções
DEVICE = None # dispositivo para executar o modelo (None para usar o dispositivo padrão), pode ser "cpu" 

PROJECT = ROOT / "runs" / "predict"
# RUN_NAME fica automaticamente, por exemplo, y26-1-db-1_1024_025_1_v1 
# ou seja, nome do modelo + tamanho da imagem + confiança + max detecções + indice para n haver overwrite. 
# index atualizo manualmente
index = 1
RUN_NAME = f"{modelo}_{IMAGE_SIZE}_{str(CONFIDENCE).replace('.', '')}_{MAX_DETECTIONS}_{index}"
#------------------------------------------------------------------------------------------------------------





def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

    run_dir = PROJECT / RUN_NAME
    if SAVE and run_dir.exists():
        raise FileExistsError(
            f"Prediction output already exists: {run_dir}. "
            "Choose a new RUN_NAME."
        )

    model = YOLO(str(MODEL_PATH))
    model.predict(
        source=SOURCE,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        max_det=MAX_DETECTIONS,
        show=SHOW,
        save=SAVE,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=False,
        device=DEVICE,
    )

    if SAVE:
        print(f"Resultados guardados em: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
