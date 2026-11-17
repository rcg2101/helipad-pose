# Modelos e execuções YOLO-Pose

## Organização dos ficheiros

- `weights/pretrained/`: pesos base, sem treino neste projeto.
- `runs/train/<experiencia>/`: resultados completos de cada treino, incluindo
  `weights/best.pt`, `weights/last.pt`, métricas, gráficos e argumentos.
- `runs/predict/<experiencia>/`: imagens/vídeos anotados de cada inferência.
- `models/`: pesos escolhidos para uso geral ou integração noutros scripts.

Os pesos de uma execução ficam junto dos resultados que os produziram. Copiar um
`best.pt` para `models/` é opcional; não apagues a execução de origem.

## Pesos e histórico

- YOLO11n-Pose pré-treino: `weights/pretrained/yolo11n-pose.pt`
- YOLO26n-Pose pré-treino: `weights/pretrained/yolo26n-pose.pt`
- YOLO26n (`weights/pretrained/yolo26n.pt`)

- YOLO26 treinado com database inicial: `runs/train/y26-1-db-1/weights/best.pt`
Notação usada: y26 é yolo26n-pose; -1 significa 1o treino realizado; db-1 significa que foi treinado como 1o database que criei
- YOLO11 treinado com database inicial: `runs/train/y11-1-db-1/weights/best.pt`



## Treinar

O treino principal está em [train_pose.py](./train_pose.py):

```powershell
python .\src\yolo\train_pose.py
```

Por omissão, usa:

- Dataset: `data/dataset/data.yaml`
- Pesos iniciais: `weights/pretrained/yolo26n-pose.pt`
- Resultados: `runs/train/y26-1-db-1/`

Antes de executar, seleciona diretamente no início de `train_pose.py` o
dataset (`DATA_YAML`), o ficheiro do modelo base (`MODEL_PATH`) e o nome da
pasta da execução (`RUN_NAME`). Por exemplo, para treinar com YOLO11, aponta
`MODEL_PATH` para `weights/pretrained/yolo11n-pose.pt` e escolhe um nome como
`y11-1-db-1`. Se a pasta já existir, o script termina sem iniciar o treino nem
substituir resultados. No fim, imprime o caminho exato para `best.pt` e
`last.pt`.

O script avançado [train_h_pose (1).py](<./Lixo%20IA/train_h_pose%20%281%29.py>)
também guarda treinos em `runs/train/` e previsões em `runs/predict/`. Para
escolher o nome de cada execução:

```powershell
python ".\src\yolo\Lixo IA\train_h_pose (1).py" train --dataset data/dataset --name y11-1-db-1
python ".\src\yolo\Lixo IA\train_h_pose (1).py" predict --weights runs/train/y26-1-db-1/weights/best.pt --source foto.jpg --name teste-foto
```

## Usar outro dataset

Em `train_pose.py`, altera `DATA_YAML` para o YAML do dataset. O YAML tem de
descrever os caminhos de `train`, `val` e, se existir, `test`, os nomes das
classes e a configuração de pose (`kpt_shape` e `flip_idx`). Os dados têm de
estar no formato YOLO-Pose; os scripts não preparam nem modificam o dataset.

## Inferência

Escolhe os pesos da experiência pretendida:

```powershell
yolo predict model="runs\train\<experiencia>\weights\best.pt" source="C:\caminho\para\a_foto.jpg" project="runs\predict" name="foto"
```

Para a webcam:

```powershell
yolo predict model="runs\train\<experiencia>\weights\best.pt" source=0 show=True project="runs\predict" name="webcam"
```

Substitui `<experiencia>` pelo nome da pasta do treino cujos pesos queres usar.
`project` define a pasta-base dos resultados e `name` identifica a subpasta
da execução; por exemplo, os resultados da imagem ficam em
`runs\predict\foto`.
