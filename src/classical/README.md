# Deteção clássica do H

`classical_h.py` abre diretamente a webcam 0, sem argumentos de terminal:

```powershell
python src\classical\classical_h.py
```

O programa usa uma única máscara adaptativa larga invertida para tinta escura, mostra o
resultado com os 12 vértices e apresenta os FPS do ciclo. Carrega em `q` para
fechar. Para usar outra câmara, altera `WEBCAM_INDEX` no início do ficheiro.

## Pipeline da webcam

1. Converte cada frame para cinzento e aplica um desfoque ligeiro.
2. Cria uma máscara adaptativa larga de tinta escura.
3. Procura o anel e o contorno do H; se o anel não for encontrado, tenta o
   contorno isolado.
4. Ajusta a homografia dos 12 vértices do modelo e escolhe a orientação em que
   o topo do H aponta mais para cima.
5. Valida a deteção pela IoU e desenha os vértices numerados.

## Exploração numa imagem

`experimentar_pipeline.py` mostra manualmente as etapas de uma imagem de
exemplo, também usando apenas a máscara adaptativa larga. Altera `IMAGE_PATH`
no início para escolher outra imagem. Carrega numa tecla para avançar entre as
etapas; nada é guardado por omissão. Para guardar as imagens, muda
`guardar_imagens = False` para `True`.