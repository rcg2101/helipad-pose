# Helipad Pose Estimation — Setup & Git Workflow

Projeto de Computer Vision para **estimação da pose de um heliponto**, comparando métodos clássicos (OpenCV + `solvePnP`) com **YOLO Pose**.

**Ambiente:** Windows + Anaconda + Git + GitHub

---

# 1. Primeiro elemento da equipa — criar o projeto

Esta parte é feita **apenas pela pessoa que cria inicialmente o ambiente e o repositório**.

## 1.1. Instalar ferramentas

Instalar:

- Anaconda/Miniconda
- Git for Windows
- Conta no GitHub

No **Anaconda Prompt**:

```bash
git config --global user.name "O Teu Nome"
git config --global user.email "o.teu@email.pt"
git config --global core.autocrlf true
```

## 1.2. Criar o ambiente

```bash
cd C:\dev

conda create -n helipad python=3.11 spyder jupyterlab -y
conda activate helipad

pip install numpy opencv-python matplotlib pandas ultralytics
```

> O ambiente Conda não vai para o Git. Apenas o `environment.yml` é partilhado.

Se houver GPU NVIDIA, confirmar:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 1.3. Criar o repositório

No GitHub:

1. **New repository**
2. Escolher um nome, por exemplo `heli-pose`
3. Adicionar `README.md`
4. Adicionar `.gitignore` para Python
5. Escolher público/privado

Clonar:

```bash
cd C:\dev
git clone https://github.com/USER/heli-pose.git
cd heli-pose
```

Criar a estrutura:

```text
heli-pose/
├── data/              # dataset — NÃO vai para o Git
├── src/
│   ├── classical/     # OpenCV + solvePnP
│   ├── yolo/          # treino e inferência YOLO
│   └── eval/          # métricas
├── notebooks/
├── environment.yml
├── .gitignore
└── README.md
```

### `.gitignore`

```gitignore
data/
runs/
*.pt
__pycache__/
.ipynb_checkpoints/
```

O dataset e os pesos treinados (`.pt`) são partilhados por **Drive/OneDrive/etc.**, não pelo Git.

### `environment.yml`

```yaml
name: helipad

channels:
  - conda-forge

dependencies:
  - python=3.11
  - spyder
  - jupyterlab
  - pip
  - pip:
      - numpy
      - opencv-python
      - matplotlib
      - pandas
      - ultralytics
```

Adicionar tudo ao Git:

```bash
git add .
git commit -m "Initial project setup"
git push
```

Finalmente, no GitHub:

**Settings → Collaborators → Add people**

Adicionar os restantes membros da equipa.

---

# 2. Restantes elementos — primeira configuração

Cada pessoa faz esta parte **uma única vez no seu próprio computador**.

## 2.1. Instalar

Instalar:

- Anaconda/Miniconda
- Git for Windows
- Conta GitHub

Configurar o Git:

```bash
git config --global user.name "O Teu Nome"
git config --global user.email "o.teu@email.pt"
git config --global core.autocrlf true
```

## 2.2. Clonar o projeto

```bash
cd C:\dev
git clone https://github.com/USER/heli-pose.git
cd heli-pose
```

## 2.3. Criar o ambiente

Cada pessoa cria o seu próprio ambiente local a partir do `environment.yml`:

```bash
conda env create -f environment.yml
conda activate helipad
```

Assim, todos trabalham com as mesmas dependências.

Para abrir as ferramentas:

```bash
conda activate helipad
spyder
```

ou:

```bash
conda activate helipad
jupyter lab
```

### Se forem adicionadas bibliotecas

Instalar localmente:

```bash
pip install scikit-learn
```

Depois **adicionar manualmente a biblioteca ao `environment.yml`** e fazer commit:

```bash
git add environment.yml
git commit -m "Add scikit-learn dependency"
git push
```

Os restantes membros atualizam o ambiente com:

```bash
conda env update -f environment.yml --prune
```

---

# 3. Fluxo diário de trabalho

Depois da configuração inicial, **não é preciso recriar o ambiente**.

## 3.1. Começar uma tarefa

```bash
conda activate helipad

git checkout main
git pull
```

Criar uma branch para a tarefa:

```bash
git checkout -b feature/deteta-H
```

Exemplos:

```text
feature/deteta-H
feature/solvePnP
feature/yolo-training
feature/evaluation
```

## 3.2. Trabalhar

Fazer as alterações normalmente.

Ver o que mudou:

```bash
git status
```

Adicionar apenas os ficheiros relevantes:

```bash
git add src/classical/detect.py
```

Fazer commit:

```bash
git commit -m "Add H detection using contours"
```

Enviar a branch:

```bash
git push -u origin feature/deteta-H
```

> `-u` só é necessário na primeira vez que a branch é enviada.

## 3.3. Pull Request

No GitHub:

1. Abrir a branch enviada.
2. Selecionar **Compare & pull request**.
3. Pedir revisão a um colega.
4. Após aprovação, fazer **Merge** para `main`.

Depois da tarefa:

```bash
git checkout main
git pull
```

E começar uma nova branch para a próxima tarefa.

---

# Regras importantes

- **Não trabalhar diretamente na `main`.**
- Uma branch por tarefa.
- Fazer commits pequenos e descritivos.
- Preferir `git add <ficheiro>` a `git add .`.
- **Nunca fazer commit do dataset ou dos pesos `.pt`.**
- Evitar que duas pessoas editem simultaneamente o mesmo notebook.
- Se houver conflitos, resolver as marcas `<<<<<<<`, `=======`, `>>>>>>>` e depois:

```bash
git add <ficheiro>
git commit
```

---

# Comandos essenciais

| Comando | Função |
|---|---|
| `git clone <URL>` | Descarregar o repositório |
| `git pull` | Atualizar com alterações da equipa |
| `git checkout -b <branch>` | Criar uma branch |
| `git status` | Ver alterações |
| `git add <ficheiro>` | Preparar alterações |
| `git commit -m "msg"` | Criar commit |
| `git push` | Enviar alterações para GitHub |
| `git log --oneline` | Ver histórico |

---

# YOLO

Exemplo de treino:

```bash
yolo train data=dataset.yaml model=yolov8n-pose.pt epochs=100
```

No Windows, se houver problemas de multiprocessing:

```bash
yolo train data=dataset.yaml model=yolov8n-pose.pt epochs=100 workers=0
```

## Armadilhas comuns

### OpenCV + Jupyter/Spyder

Preferir:

```python
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

em vez de:

```python
cv2.imshow(...)
```

O OpenCV usa **BGR**, enquanto o Matplotlib espera **RGB**.

### Caminhos

Usar `pathlib` para manter compatibilidade entre Windows e Linux:

```python
from pathlib import Path
import cv2

img = cv2.imread(
    str(Path("data") / "images" / "helipad.jpg")
)
```

---

# Organização dos ficheiros

```text
Código       → GitHub
Environment  → environment.yml
Dataset      → Drive/OneDrive
Pesos .pt    → Drive/OneDrive
Resultados   → GitHub apenas se forem pequenos/relevantes
```
