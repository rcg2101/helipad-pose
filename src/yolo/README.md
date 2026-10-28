# Treino YOLO-Pose

O treino está em [train_pose.py](./train_pose.py). Abre esse ficheiro no VS Code
ou Spyder e executa-o como um script Python normal (por exemplo, **Run Python
File** no VS Code). Não é necessário executar comandos YOLO no terminal. O
script usa o `data.yaml` existente; não o cria nem altera.

Por omissão, usa:

- Dataset: `data/dataset/data.yaml`
- Pesos iniciais: `yolo11n-pose.pt` na raiz do projeto
- Parâmetros de treino: os mesmos definidos no treino de pose anterior
- Resultados: `runs/pose/`

## Usar outro dataset

No início de `train_pose.py`, altera `DATA_YAML` para o caminho do YAML desse
dataset. Por exemplo:

```python
DATA_YAML = ROOT / "data" / "outro_dataset" / "data.yaml"
```

O YAML tem de descrever os caminhos de `train`, `val` e, se existir, `test`,
além dos nomes das classes e da configuração de pose (`kpt_shape` e `flip_idx`).
As imagens e labels também têm de estar no formato YOLO-Pose compatível com o
modelo. O script não prepara nem modifica o dataset.

## Preservar e comparar treinos

Cada execução é guardada numa pasta própria dentro de `runs/pose/`. O script
usa `exist_ok=False`: se `h12` já existir, o Ultralytics cria uma nova pasta
numerada em vez de substituir a anterior. Assim, os pesos antigos continuam
disponíveis para comparar ou usar mais tarde.

Em cada pasta do treino:

- `weights/best.pt` contém os pesos com melhor resultado na validação;
- `weights/last.pt` contém os pesos da última época.

O script imprime os caminhos exatos no fim do treino. Para comparar de forma
justa, mantém iguais o dataset e os parâmetros e altera apenas o que queres
testar. Não apagues as pastas antigas de `runs/pose/`.

Para fazer inferência, seleciona o `best.pt` da execução pretendida como os
pesos do modelo, por exemplo no comando YOLO:

```powershell
yolo predict model="runs/pose/h12/weights/best.pt" source="C:\caminho\para\a_foto.jpg"
```
