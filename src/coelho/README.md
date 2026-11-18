"""
calibracao1.py: calibração da câmara com um tabuleiro de xadrez.
 
O QUE FAZ
    Lê as fotos do tabuleiro em data/calib/, deteta os cantos internos em cada
    uma e calcula:
      - K     : matriz intrínseca (distâncias focais fx, fy e centro ótico cx, cy, em px)
      - dist  : coeficientes de distorção da lente (k1, k2, p1, p2, k3)
    Guarda tudo em camera.npz, que os restantes módulos (PnP, eixos, métricas)
    carregam. Sem K não é possível converter píxeis em metros.
 
BASE
    Código do tutorial oficial de calibração do OpenCV (Camera Calibration, Python),
    com adaptações assinaladas nos comentários. Foi escrito com ajuda de IA
    generativa (ver README, secção "Uso de IA generativa").
 
COMO CORRER
    %runfile calibracao1.py --wdir   (no Spyder)   ou   python calibracao1.py
 
ENTRADAS
    data/calib/*.jpg|png   fotos do tabuleiro, todas com a MESMA resolução e orientação
SAÍDAS
    src/coelho/camera.npz  com K, dist e size (largura, altura)
 
NOTA IMPORTANTE
    O K só é válido para a mesma câmara, resolução e orientação das fotos de
    calibração (aqui: telemóvel, 1200x1600 em vertical). Se a resolução mudar,
    é preciso recalibrar ou escalar o K.
"""