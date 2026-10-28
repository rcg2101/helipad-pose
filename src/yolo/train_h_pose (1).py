#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_h_pose.py — Treino e avaliação de YOLO11n-pose para os 12 vértices de um "H"
=================================================================================

Subcomandos (todos com --help):

    python train_h_pose.py validate --dataset dataset
    python train_h_pose.py train    --dataset dataset --epochs 150
    python train_h_pose.py eval     --weights runs/pose/h12/weights/best.pt --dataset dataset --split val
    python train_h_pose.py predict  --weights runs/pose/h12/weights/best.pt --source foto.jpg
    python train_h_pose.py split    --raw raw --out dataset --seed 0     (só para imagens REAIS)

Dependências:  pip install ultralytics pyyaml numpy
O `eval` com pose 3D usa funções do gen_synthetic.py (mesma pasta) e o meta.json.

-----------------------------------------------------------------------------
1. O QUE TENS DE TER NA BASE DE DADOS
-----------------------------------------------------------------------------

1.1 Ficheiros
    Uma imagem + um .txt com o MESMO nome, em pastas espelhadas:

        dataset/
          images/train/  ...      labels/train/  ...      poses/train/  ...
          images/val/    ...      labels/val/    ...      poses/val/    ...
          images/test/   ...      labels/test/   ...      poses/test/   ...
          meta.json

    O Ultralytics encontra o label trocando "images" por "labels" no caminho.
    Imagens sem H (negativos): .txt vazio ou inexistente (o gerador cria vazios).
    `poses/` e `meta.json` não são usados no treino; servem para o `eval` medir
    o erro da pose 3D (secção 1.8). Com o gerador, copia também o meta.json:
        Copy-Item synth\\meta.json dataset\\

1.2 Formato do .txt (uma linha por H; 41 números separados por espaço)

        class cx cy w h  x0 y0 v0  x1 y1 v1  ...  x11 y11 v11

    - class : 0 (só existe a classe "H")
    - cx cy w h : caixa do H (centro e dimensões), normalizados [0,1]
                  (x / largura, y / altura). Caixa justa à volta do H, sem o anel.
    - xi yi : posição do vértice i em píxeis / (largura, altura), em [0,1].
    - vi    : 0 = fora da imagem / não anotado (x=y=0),
              1 = dentro da imagem mas oculto, 2 = visível.

1.3 Numeração dos 12 vértices (convenção fixa, presa ao objeto físico; contorno
    em sentido horário a partir do canto superior esquerdo, H "de pé"):

      0 ───── 1            4 ───── 5
      │       │            │       │
      │       2 ────────── 3       │
      │                            │
      │       9 ────────── 8       │
      │       │            │       │
     11 ───── 10           7 ───── 6

    Espelhamento horizontal (fliplr): 0<->5, 1<->4, 2<->3, 6<->11, 7<->10, 8<->9
    (FLIP_IDX abaixo). Se mudares a numeração, recalcula o FLIP_IDX ou usa fliplr=0.

1.4 Ambiguidade de 180° (yaw)
    O H rodado 180° é igual. Por isso o dataset limita o yaw a +-80° (gerador) e o
    treino NÃO roda as imagens (degrees=0 por defeito): rodar mais empurraria o
    yaw acima de 90° e reintroduziria exemplos idênticos com labels diferentes.
    Também flipud=0 (o Ultralytics não aplica flip_idx na vertical).

1.5 Cobertura: várias distâncias (H pequeno e grande), roll/pitch/yaw, luz, fundos,
    H parcialmente fora da imagem ou ocluído. Mais variedade vale mais do que
    mais epochs. O gerador sintético (gen_synthetic.py) cobre isto para o treino.

1.6 Divisão train/val/test
    - O treino precisa de images/val (o Ultralytics falha sem ela); o test só é
      necessário no `eval --split test`.
    - Val e test devem ser REAIS e de sessões diferentes do treino. Um val
      sintético serve só para testar a pipeline: as métricas saem otimistas
      (o treino avisa se o val parecer sintético).
    - `split` divide imagens reais por sessão (raw/images/<sessao>/*.jpg e
      raw/labels/<sessao>/*.txt); frames consecutivos nunca se repartem.

1.7 Anotação de imagens reais
    CVAT, Roboflow ou Label Studio (exportar em "YOLO pose"/"COCO keypoints"); COCO
    converte-se com ultralytics.data.converter.convert_coco(..., use_keypoints=True).
    Confirma com `validate`.

1.8 Pose de referência (para o `eval` em metros e graus)
    poses/<split>/<nome>.json com pelo menos "image", "R" (3x3) e "t_m" (3):
    X_cam = R @ X_alvo + t (convenção do OpenCV, alvo -> câmara), e meta.json com
    "K", "image_size" e "vertices_3d_m". O gerador escreve ambos. Em imagens reais
    é a pose de referência (ex.: calculada por ArUco), no mesmo formato.

-----------------------------------------------------------------------------
2. COMO O TREINO ESTÁ CONFIGURADO (e porquê)
-----------------------------------------------------------------------------
    - Parte de yolo11n-pose.pt (COCO-pose, 17 keypoints): como kpt_shape é [12,3],
      a cabeça de keypoints é reconstruída e o resto (backbone/neck) é reaproveitado.
    - imgsz: o H deve ter >~35 px de largura na entrada da rede à distância máxima
      de operação (ver gen_synthetic.py verify). Treino e inferência com o mesmo imgsz.
    - Augmentations conservadoras: fliplr=0.5 (com flip_idx), flipud=0, degrees=0,
      scale=0.5, translate=0.1, HSV, mosaic (desligado nas últimas 10 epochs).
    - patience=30: pára se o val não melhorar durante 30 epochs.

-----------------------------------------------------------------------------
3. REFERÊNCIAS
-----------------------------------------------------------------------------
 [1] Ultralytics — Pose estimation (tarefa e métricas)
     https://docs.ultralytics.com/tasks/pose/
 [2] Ultralytics — Formato de dataset de pose (kpt_shape, flip_idx, labels)
     https://docs.ultralytics.com/datasets/pose/
 [3] Ultralytics — Modo train (argumentos e augmentations)
     https://docs.ultralytics.com/modes/train/
 [4] Ultralytics — YOLO11
     https://docs.ultralytics.com/models/yolo11/
 [5] Maji et al., "YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation
     Using Object Keypoint Similarity Loss", CVPR Workshops, 2022 (arXiv:2204.06806)
 [6] COCO — Avaliação de keypoints (OKS)
     https://cocodataset.org/#keypoints-eval
 [7] Lepetit, Moreno-Noguer, Fua, "EPnP: An Accurate O(n) Solution to the PnP
     Problem", IJCV 81, 2009
 [8] Collins & Bartoli, "Infinitesimal Plane-Based Pose Estimation", IJCV 109, 2014
     (IPPE, usado em cv2.SOLVEPNP_IPPE para alvos planares)
 [9] OpenCV — solvePnP
     https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

# ----------------------------------------------------------------------------
# Constantes do problema (ver secção 1.3 do cabeçalho)
# ----------------------------------------------------------------------------
NUM_KPTS = 12
KPT_DIM = 3  # x, y, visibilidade
CLASS_NAMES = {0: "H"}

# Permutação dos keypoints sob espelhamento horizontal.
FLIP_IDX = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6]

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


def label_path_for(img: Path) -> Path:
    """Caminho do label de uma imagem (troca o último componente 'images' por 'labels')."""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def parse_label_line(line: str):
    """Devolve (cls, bbox[4], kpts[12,3]) a partir de uma linha de label."""
    vals = np.array(line.split(), dtype=float)
    if vals.size != 5 + NUM_KPTS * KPT_DIM:
        raise ValueError(f"esperados {5 + NUM_KPTS * KPT_DIM} valores, encontrados {vals.size}")
    return int(vals[0]), vals[1:5], vals[5:].reshape(NUM_KPTS, KPT_DIM)


# ----------------------------------------------------------------------------
# data.yaml
# ----------------------------------------------------------------------------
def write_data_yaml(dataset: Path) -> Path:
    cfg = {
        "path": str(dataset.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [NUM_KPTS, KPT_DIM],
        "flip_idx": FLIP_IDX,
        "names": CLASS_NAMES,
    }
    out = dataset / "data.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


# ----------------------------------------------------------------------------
# validate: verifica o dataset antes de gastar tempo de GPU
# ----------------------------------------------------------------------------
def cmd_validate(args):
    dataset = Path(args.dataset)
    problems = 0

    # O flip_idx tem de ser uma involução (aplicar duas vezes volta ao início).
    if sorted(FLIP_IDX) != list(range(NUM_KPTS)) or any(FLIP_IDX[FLIP_IDX[i]] != i for i in range(NUM_KPTS)):
        print("ERRO: FLIP_IDX não é uma permutação válida/involutiva.")
        problems += 1

    for split in SPLITS:
        img_dir = dataset / "images" / split
        if not img_dir.exists():
            print(f"[{split}] pasta não existe, a saltar")
            continue
        imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
        n_pos = n_neg = n_bad = 0
        for img in imgs:
            lbl = label_path_for(img)
            if not lbl.exists():
                n_neg += 1  # sem label = imagem de fundo
                continue
            lines = [l for l in lbl.read_text().splitlines() if l.strip()]
            if not lines:
                n_neg += 1
                continue
            n_pos += 1
            for ln, line in enumerate(lines, 1):
                try:
                    cls, box, kp = parse_label_line(line)
                    assert cls == 0, f"classe {cls} (esperada 0)"
                    assert np.all((box >= 0) & (box <= 1)), "bbox fora de [0,1]"
                    assert np.all((kp[:, :2] >= 0) & (kp[:, :2] <= 1)), "keypoint fora de [0,1]"
                    assert set(np.unique(kp[:, 2])) <= {0.0, 1.0, 2.0}, "visibilidade fora de {0,1,2}"
                    vis = kp[:, 2] > 0
                    if vis.any():
                        x0, y0 = box[0] - box[2] / 2, box[1] - box[3] / 2
                        x1, y1 = box[0] + box[2] / 2, box[1] + box[3] / 2
                        tol = 0.01
                        inside = (
                            (kp[vis, 0] >= x0 - tol) & (kp[vis, 0] <= x1 + tol)
                            & (kp[vis, 1] >= y0 - tol) & (kp[vis, 1] <= y1 + tol)
                        )
                        assert inside.all(), "keypoints fora da bbox (verifica a anotação)"
                except Exception as e:  # noqa: BLE001
                    print(f"  {lbl}:{ln}: {e}")
                    n_bad += 1
        problems += n_bad
        print(f"[{split}] {len(imgs)} imagens | {n_pos} com H | {n_neg} sem H | {n_bad} linhas com erro")

    print("OK" if problems == 0 else f"{problems} problema(s) encontrado(s)")
    sys.exit(1 if problems else 0)


# ----------------------------------------------------------------------------
# split: train/val/test por sessão, para imagens REAIS (ver 1.6)
# ----------------------------------------------------------------------------
def cmd_split(args):
    raw, out = Path(args.raw), Path(args.out)
    img_root, lbl_root = raw / "images", raw / "labels"
    ratios = dict(zip(SPLITS, (args.train, args.val, args.test)))
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        sys.exit("As proporções têm de somar 1.")

    # Agrupar por sessão (primeira subpasta). Sem subpastas: cada imagem é uma
    # "sessão" própria, o que NÃO evita a fuga de dados entre frames vizinhos.
    groups = {}
    flat = False
    for img in sorted(img_root.rglob("*")):
        if img.suffix.lower() not in IMG_EXT:
            continue
        rel = img.relative_to(img_root)
        lbl = (lbl_root / rel).with_suffix(".txt")
        if not lbl.exists():
            print(f"aviso: sem label para {img}, ignorada (cria um .txt vazio se for imagem de fundo)")
            continue
        if len(rel.parts) > 1:
            session = rel.parts[0]
        else:
            session, flat = rel.stem, True
        groups.setdefault(session, []).append((img, lbl))
    if flat:
        print("AVISO: imagens sem subpastas de sessão; o split será por imagem (risco de fuga de dados).")

    total = sum(len(v) for v in groups.values())
    if total == 0:
        sys.exit("Nenhuma imagem com label encontrada.")

    # Atribuição gulosa: cada sessão (em ordem aleatória reprodutível) vai para
    # o split com maior défice relativamente ao alvo.
    sessions = sorted(groups)
    random.Random(args.seed).shuffle(sessions)
    count = {s: 0 for s in SPLITS}
    assign = {}
    for sess in sessions:
        deficit = {s: ratios[s] * total - count[s] for s in SPLITS if ratios[s] > 0}
        best = max(deficit, key=deficit.get)
        assign[sess] = best
        count[best] += len(groups[sess])

    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for sess, split in assign.items():
        for img, lbl in groups[sess]:
            name = f"{sess}__{img.stem}"  # evita colisões de nomes entre sessões
            shutil.copy2(img, out / "images" / split / f"{name}{img.suffix}")  # copia, não move
            shutil.copy2(lbl, out / "labels" / split / f"{name}.txt")

    yaml_path = write_data_yaml(out)
    print(f"Split (seed={args.seed}): " + ", ".join(f"{s}={count[s]}" for s in SPLITS))
    for s in SPLITS:
        print(f"  {s}: " + ", ".join(sorted(k for k, v in assign.items() if v == s)))
    print(f"data.yaml escrito em {yaml_path}")
    print("NOTA: o split não copia poses/ nem meta.json; copia-os à mão se quiseres o eval em metros e graus.")


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
def warn_if_synthetic_val(dataset: Path):
    val = dataset / "images" / "val"
    if not val.exists():
        sys.exit(f"Falta {val}: o Ultralytics precisa de um conjunto de validação para treinar "
                 "(ver 1.6 do cabeçalho).")
    names = [p.name for p in val.rglob("*") if p.suffix.lower() in IMG_EXT]
    if not names:
        sys.exit(f"{val} está vazia.")
    if all(n.startswith("synth") for n in names):
        print("AVISO: o val só tem imagens sintéticas. As métricas saem otimistas e o best.pt é escolhido "
              "sobre elas; serve para testar a pipeline, não para reportar. Troca por imagens reais "
              "assim que as tiveres.\n")


def cmd_train(args):
    from ultralytics import YOLO

    dataset = Path(args.dataset)
    data_yaml = write_data_yaml(dataset)  # regenera sempre, para ficar coerente com as constantes acima
    warn_if_synthetic_val(dataset)

    if args.resume:
        # Continuar um treino interrompido: usa o last.pt dessa corrida.
        YOLO(args.resume).train(resume=True)
        return

    # Carrega pesos pré-treinados no COCO-pose (17 keypoints). Como kpt_shape é
    # [12,3], o Ultralytics reconstrói a cabeça de keypoints e reaproveita o
    # resto (backbone/neck) — é transfer learning.
    model = YOLO(args.model)

    kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,   # early stopping: pára se o val não melhorar
        imgsz=args.imgsz,         # H com >~35 px na entrada à distância máxima (ver cabeçalho)
        batch=args.batch,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # --- augmentations (ver 1.4): conservadoras por causa da simetria do H
        fliplr=0.5,               # ok porque o H é simétrico e FLIP_IDX está definido
        flipud=0.0,               # NUNCA: o Ultralytics não aplica flip_idx na vertical
        degrees=args.degrees,     # 0: o dataset já cobre o yaw; rodar mais reintroduz a ambiguidade de 180°
        scale=0.5,                # variação de tamanho do H
        translate=0.1,
        perspective=0.0,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,  # variação de cor/luz
        mosaic=1.0,
        close_mosaic=10,          # desliga o mosaic nas últimas epochs
    )
    if args.device:
        kwargs["device"] = args.device
    if args.workers is not None:
        kwargs["workers"] = args.workers  # no Windows, se bloquear/der erro de memória: 0 ou 2
    else:
        workers_val = 0
    if args.cache:
        kwargs["cache"] = True            # imagens em RAM: mais rápido, usa memória
    model.train(**kwargs)
    print(f"Pesos em: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


# ----------------------------------------------------------------------------
# eval: mAP de pose + erro por keypoint (px) + erro da pose 3D (m, graus)
# ----------------------------------------------------------------------------
def load_geom():
    """Funções de pose do gen_synthetic.py (mesma pasta); None se não existir."""
    try:
        import gen_synthetic as g
        return g
    except ImportError:
        return None


def summarize_pose(rows):
    """rows: dicts com d, e_t (mm), e_rot (graus), att (roll,pitch,yaw erros), ok."""
    n = len(rows)
    good = [r for r in rows if r["ok"]]
    print(f"\nPOSE 3D (PnP sobre os keypoints previstos) — {n} imagens com pose de referência")
    print(f"  sem deteção: {sum(r['status'] == 'miss' for r in rows)} | "
          f"PnP impossível (<4 keypoints fiáveis): {sum(r['status'] == 'fewkp' for r in rows)} | "
          f"com pose: {len(good)}")
    if not good:
        return
    et = np.array([r["e_t"] for r in good])
    rel = np.array([100 * r["e_t"] / 1000 / r["d"] for r in good])
    er = np.array([r["e_rot"] for r in good])
    flips = np.mean(er > 15) * 100
    print(f"  translação: mediana {np.median(et):.1f} mm ({np.median(rel):.2f}% da distância), "
          f"p95 {np.percentile(et, 95):.1f} mm")
    print(f"  rotação:    mediana {np.median(er):.2f}°, p95 {np.percentile(er, 95):.2f}° | "
          f"solução errada do PnP plano (erro >15°): {flips:.1f}%")
    att = np.array([r["att"] for r in good])
    print(f"  atitude (mediana |erro|): roll {np.median(np.abs(att[:, 0])):.2f}°, "
          f"pitch {np.median(np.abs(att[:, 1])):.2f}°, yaw {np.median(np.abs(att[:, 2])):.2f}°")
    d = np.array([r["d"] for r in good])
    if len(good) >= 8:
        edges = np.quantile(d, np.linspace(0, 1, 5))
        print("  por distância:   d (m)         |   n | med|dt| mm | med dt/d % | med rot° | flip %")
        for a, b in zip(edges[:-1], edges[1:]):
            m = (d >= a) & (d <= b)
            if m.sum() == 0:
                continue
            print(f"                   {a:5.2f} .. {b:5.2f} | {m.sum():3d} | {np.median(et[m]):10.1f} | "
                  f"{np.median(rel[m]):10.2f} | {np.median(er[m]):8.2f} | {np.mean(er[m] > 15) * 100:6.1f}")


def cmd_eval(args):
    from ultralytics import YOLO

    dataset = Path(args.dataset)
    data_yaml = write_data_yaml(dataset)
    model = YOLO(args.weights)

    # (a) métricas padrão do Ultralytics (caixa e pose/OKS).
    # Nota: com kpt_shape diferente de [17,3] o OKS usa sigmas uniformes, por
    # isso o mAP de pose não é comparável com o do COCO.
    m = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, plots=args.plots,
                  **({"device": args.device} if args.device else {}))
    print(f"box  mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")
    print(f"pose mAP50={m.pose.map50:.3f}  mAP50-95={m.pose.map:.3f}")

    # (b) erro em píxeis por keypoint e (c) erro da pose 3D, com a deteção de maior confiança.
    g = None if args.no_pose else load_geom()
    meta = None
    meta_path = Path(args.meta) if args.meta else dataset / "meta.json"
    poses_dir = dataset / "poses" / args.split
    if not args.no_pose:
        if g is None:
            print("\nAVISO: sem gen_synthetic.py na pasta; salto a pose 3D.")
        elif not meta_path.exists() or not poses_dir.exists():
            print(f"\nNota: sem {meta_path} ou {poses_dir}; salto a pose 3D (ver 1.8 do cabeçalho).")
        else:
            meta = json.loads(meta_path.read_text())
    if meta is not None:
        K = np.array(meta["K"], dtype=np.float64)
        Pobj = np.array(meta["vertices_3d_m"], dtype=np.float64)

    img_dir = dataset / "images" / args.split
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    errs = [[] for _ in range(NUM_KPTS)]
    n_gt = n_miss = 0
    pose_rows = []
    for img in imgs:
        lbl = label_path_for(img)
        has_gt = lbl.exists() and lbl.read_text().strip() != ""
        pj = poses_dir / f"{img.stem}.json"
        if not has_gt and not (meta is not None and pj.exists()):
            continue
        res = model.predict(str(img), imgsz=args.imgsz, conf=args.conf, verbose=False,
                            **({"device": args.device} if args.device else {}))[0]
        detected = res.boxes is not None and len(res.boxes) > 0
        if detected:
            best = int(res.boxes.conf.argmax())
            kxy = res.keypoints.xy[best].cpu().numpy()  # (12,2) em píxeis da imagem original
            kc = (res.keypoints.conf[best].cpu().numpy() if res.keypoints.conf is not None
                  else np.ones(NUM_KPTS))
            h, w = res.orig_shape

        if has_gt:
            _, _, gt = parse_label_line(lbl.read_text().splitlines()[0])
            n_gt += 1
            if not detected:
                n_miss += 1
            else:
                gt_xy = gt[:, :2] * np.array([w, h])
                for k in range(NUM_KPTS):
                    if gt[k, 2] > 0:  # só keypoints anotados
                        errs[k].append(float(np.linalg.norm(kxy[k] - gt_xy[k])))

        if meta is not None and pj.exists():
            js = json.loads(pj.read_text())
            Rg, tg = np.array(js["R"]), np.array(js["t_m"])
            row = {"d": float(np.linalg.norm(tg)), "ok": False, "status": "ok"}
            if not detected:
                row["status"] = "miss"
            else:
                use = kc >= args.kconf
                if use.sum() < 4:
                    row["status"] = "fewkp"
                else:
                    sol = g.solve_pose(kxy[use], K, None, Pobj[use])
                    if sol is None:
                        row["status"] = "fewkp"
                    else:
                        rv, tv = sol
                    if sol is not None:
                        import cv2
                        Re = cv2.Rodrigues(rv)[0]
                        ae, ag = g.R_to_att(Re), g.R_to_att(Rg)
                        row.update(ok=True, e_t=float(np.linalg.norm(tv - tg) * 1000),
                                   e_rot=g.rot_err_deg(Re, Rg),
                                   att=[g.wrap180(ae[k] - ag[k]) for k in ("roll", "pitch", "yaw")])
            pose_rows.append(row)

    print(f"\nKEYPOINTS (px) — imagens com H: {n_gt} | sem deteção: {n_miss} ({100 * n_miss / max(n_gt, 1):.1f}%)")
    print("kpt |    n | média px | mediana px | p95 px")
    for k, e in enumerate(errs):
        if e:
            print(f"{k:3d} | {len(e):4d} | {np.mean(e):8.2f} | {np.median(e):10.2f} | {np.percentile(e, 95):6.2f}")
        else:
            print(f"{k:3d} | sem dados")
    if meta is not None:
        summarize_pose(pose_rows)


# ----------------------------------------------------------------------------
# predict: guarda imagens/vídeos anotados (caixa + esqueleto de keypoints)
# ----------------------------------------------------------------------------
def cmd_predict(args):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, save=True,
                  project=args.project, name="predict", exist_ok=True)
    print(f"Resultados em {Path(args.project) / 'predict'}")


# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="YOLO11n-pose para os 12 vértices de um H")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="verifica labels do dataset")
    v.add_argument("--dataset", default="dataset")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("split", help="divide imagens reais em train/val/test por sessão")
    s.add_argument("--raw", default="raw")
    s.add_argument("--out", default="dataset")
    s.add_argument("--train", type=float, default=0.70)
    s.add_argument("--val", type=float, default=0.15)
    s.add_argument("--test", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_split)

    t = sub.add_parser("train", help="treina o modelo")
    t.add_argument("--dataset", default="dataset")
    t.add_argument("--model", default="yolo11n-pose.pt")
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--patience", type=int, default=30)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--degrees", type=float, default=0.0, help="rotação aleatória (graus); 0 por causa do yaw (ver 1.4)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None, help="ex.: 0 (GPU), cpu, mps")
    t.add_argument("--workers", type=int, default=None, help="processos de leitura de dados (Windows: 0-4)")
    t.add_argument("--cache", action="store_true", help="guarda as imagens em RAM")
    t.add_argument("--resume", default=None, help="caminho para um last.pt para continuar um treino")
    t.add_argument("--project", default="runs/pose")
    t.add_argument("--name", default="h12")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="mAP, erro por keypoint (px) e erro da pose 3D (m, graus)")
    e.add_argument("--weights", required=True)
    e.add_argument("--dataset", default="dataset")
    e.add_argument("--split", default="val", choices=SPLITS)
    e.add_argument("--imgsz", type=int, default=640)
    e.add_argument("--conf", type=float, default=0.25)
    e.add_argument("--kconf", type=float, default=0.5, help="confiança mínima de cada keypoint para entrar no PnP")
    e.add_argument("--meta", default=None, help="meta.json (por defeito DATASET/meta.json)")
    e.add_argument("--no_pose", action="store_true", help="salta a avaliação da pose 3D")
    e.add_argument("--plots", action="store_true", help="guarda os gráficos do Ultralytics")
    e.add_argument("--device", default=None)
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("predict", help="inferência em imagem/pasta/vídeo")
    r.add_argument("--weights", required=True)
    r.add_argument("--source", required=True)
    r.add_argument("--imgsz", type=int, default=640)
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--project", default="runs/pose")
    r.set_defaults(func=cmd_predict)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
