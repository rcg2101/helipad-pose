# Deteção clássica do H

O script `classical_h.py` deteta o H por segmentação e contornos, sem YOLO. A
janela mostra um mosaico com a imagem original, imagem em cinzento, máscaras
Otsu/adaptativas, contornos, máscara do H e resultado.

## Pipeline

1. Converte cada frame para cinzento e aplica um desfoque ligeiro.
2. Cria máscaras binárias com Otsu e limiarização adaptativa para tinta escura
   ou clara.
3. Procura o anel nos contornos e valida a sua forma elíptica, proporção e
   concentricidade.
4. Procura o H dentro do anel; se não encontrar um anel, tenta detetar o H pelo
   contorno isolado.
5. Ajusta uma homografia entre os 12 vértices do modelo e o contorno. Como o H
   é simétrico a 180°, escolhe a orientação em que o topo aponta mais para cima
   na imagem.
6. Calcula a sobreposição entre a máscara prevista e a segmentada; aceita a
   deteção se a IoU ultrapassar o limiar configurado.
7. Mostra o original, as máscaras, os contornos e o resultado com os vértices
   numerados.

Executar a partir da raiz do projeto:

```powershell
python src/classical/classical_h.py image "C:\caminho\para\foto.jpg"
python src/classical/classical_h.py image data/dataset/images/val
python src/classical/classical_h.py video --source 0
python src/classical/classical_h.py video --source "C:\caminho\para\video.mp4"
```

Os mosaicos de imagens são guardados por omissão em
`src/classical/classical_out/`. Para escolher outra pasta, usa
`--save-dir "C:\caminho\para\pasta"`. A visualização de webcam/vídeo não grava
ficheiros.

Na janela:

- `m` ou `0`: mosaico de todas as fases;
- `1` a `9`: abre uma fase/máscara individual;
- `]` ou `n`: próxima fase (incluindo as fases depois da 9);
- `[` ou `b`: fase anterior;
- em imagens, outra tecla avança para a próxima imagem;
- `q`: sair.

O mosaico ajusta o tamanho das fases à resolução do ecrã e identifica cada
imagem com o número do passo correspondente.

As opções `--polarity dark|light|auto`, `--blur`, `--min-ring-px`,
`--min-h-area` e `--min-iou` permitem ajustar a segmentação. `auto` testa
tinta escura e clara. O valor por omissão de `--min-iou` é 0.80.