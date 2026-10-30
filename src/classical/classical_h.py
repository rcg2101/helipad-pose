#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classical_h.py — Deteção dos 12 vértices do H por visão clássica (sem rede neuronal)
====================================================================================

Devolve os 12 keypoints na MESMA ordem e formato da rede YOLO-pose (ordem de
H_UNITS no gen_synthetic.py), por isso o PnP, o webcam_test.py e o eval servem
sem alterações.

Pipeline (cada fase aparece num "tile" do mosaico de debug)
-----------------------------------------------------------
 1. original
 2. cinzento + desfoque gaussiano
 3. binarização (tinta = escuro). Tenta Otsu e 2 limiares adaptativos (blocos
    grande e pequeno), porque o tamanho certo do bloco depende da distância.
 4. contornos (findContours RETR_CCOMP: contornos exteriores + buracos).
    Procura o ANEL: contorno com um buraco, ambos elipses (fitEllipse com
    resíduo baixo), quase concêntricos e com razão de áreas ~ (4/5)^2.
    Sob qualquer perspetiva um círculo é uma elipse, por isso este teste é
    exato e muito seletivo (um H sozinho ou uma letra nunca passa).
 5. anel + H: o H é o contorno exterior dentro do buraco do anel, com
    área ~24% da do buraco e solidez ~0.6 (12 x² / 20 x², invariante à perspetiva).
 6. frame canónico: a elipse do buraco (raio 4x) dá centro, escala e
    achatamento; falta só a rotação no plano. Procura-se o ângulo (0..180°) que
    maximiza o IoU entre a máscara do H e o modelo girado (96x96 px).
 7. ICP ponto-a-reta (binário): ajusta a homografia modelo->imagem aos
    contornos do H (todas as arestas, não só os cantos).
 8. refinamento sub-pixel em níveis de cinzento: ao longo da normal de cada
    aresta procura o máximo do gradiente e reajusta a homografia.
 9. resultado: 12 vértices = homografia aplicada aos vértices do modelo (a
    ordem vem do modelo, não da geometria na imagem, por isso não há problemas
    de "top-left/top-right" com yaw ou perspetiva).

Se o anel não aparece (cortado, oculto, imagem muito pequena) há um modo de
recurso "h_only": contorno do H com approxPolyDP a 12 vértices + correspondência
cíclica com o modelo.

Ambiguidade de 180°: o H é simétrico por rotação de 180°. Escolhe-se a hipótese em
que o "topo" do H aponta mais para cima na imagem (câmara com topo da imagem =
frente do drone => |yaw| < 90°). Se o H está quase de lado (|ângulo| > --yaw_max)
marca-se `flip_ambig`.

Uso
---
  # uma imagem (ou pasta): grava o mosaico com todas as fases em classical_out/
  python classical_h.py image foto.jpg
  python classical_h.py image synth/images/val --out_dir out --show

  # tempo real (webcam), com mosaico de fases; teclas 0 = mosaico, 1..9 = fase
  python classical_h.py webcam --calib calib.npz
  python classical_h.py webcam --source video.mp4 --view 9

  # medir contra o dataset sintético (erro em px, taxa de deteção, falsos positivos)
  python classical_h.py test --data synth --split val --save_fail 8

Dependências: pip install opencv-python numpy   (+ gen_synthetic.py e webcam_test.py ao lado)
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from gen_synthetic import H_UNITS, R_to_att, obj_pts, rot_err_deg, solve_pose, project
from webcam_test import IMG_EXT, Source, add_cam_args, draw_pose, estimate_pose, prepare_K, put_text

NK = len(H_UNITS)  # 12
STAGE_NAMES = ["1 original", "2 cinzento+blur", "3 binaria", "4 contornos / aneis", "5 anel + H",
               "6 frame canonico", "7 ICP binario", "8 arestas (cinzento)", "9 resultado"]


# ----------------------------------------------------------------------------
# Parâmetros
# ----------------------------------------------------------------------------
def default_params(**kw):
    p = dict(
        polarity="dark",       # tinta escura em papel claro (dark | light | auto)
        blur=1.0,              # sigma do desfoque gaussiano (px)
        min_ring_px=24,        # eixo menor mínimo do anel exterior (px)
        min_axis_ratio=0.25,   # eixo menor/maior mínimo do anel (inclinação máx. ~75°)
        min_iou=0.80,          # IoU mínimo (modelo projetado vs máscara do H) para aceitar
        yaw_max=80.0,          # |ângulo do topo do H| acima disto => flip_ambig
        max_ring_cand=12,      # nº máx. de candidatos a anel por máscara
        gray_refine=True,      # refinamento sub-pixel com gradiente
        h_only=True,           # modo de recurso sem anel
        min_h_area=60,         # área mínima do H no modo h_only (px²)
    )
    p.update(kw)
    return SimpleNamespace(**p)


# ----------------------------------------------------------------------------
# Geometria do modelo
# ----------------------------------------------------------------------------
def apply_h(h, pts):
    p = np.c_[np.asarray(pts, np.float64), np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3]


def scale_of(h):
    """Píxeis por unidade x no centro do modelo."""
    p = apply_h(h, np.array([[0, 0], [1, 0], [0, 1]], np.float64))
    a, b = p[1] - p[0], p[2] - p[0]
    return math.sqrt(abs(a[0] * b[1] - a[1] * b[0]))


def build_model_samples(step=0.05):
    """Pontos densos sobre o contorno do H (unidades x), normais exteriores e distância ao vértice."""
    poly = H_UNITS.astype(np.float32).reshape(-1, 1, 2)
    pts, nrm, dv = [], [], []
    for k in range(NK):
        a, b = H_UNITS[k], H_UNITS[(k + 1) % NK]
        d = b - a
        L = float(np.linalg.norm(d))
        t = d / L
        n = np.array([t[1], -t[0]])
        mid = (a + b) / 2
        if cv2.pointPolygonTest(poly, (float(mid[0] + 0.05 * n[0]), float(mid[1] + 0.05 * n[1])), False) > 0:
            n = -n
        m = max(2, int(round(L / step)))
        for s in (np.arange(m) + 0.5) / m * L:
            pts.append(a + t * s)
            nrm.append(n)
            dv.append(min(s, L - s))
    return SimpleNamespace(pts=np.array(pts), nrm=np.array(nrm), dv=np.array(dv))


MS = build_model_samples()
_TPL = None


def templates():
    """Modelo do H rasterizado (96x96, 8 px/unidade) para ângulos 0..178° de 2 em 2."""
    global _TPL
    if _TPL is None:
        angs = np.arange(0, 180, 2.0)
        tpl = []
        for a in angs:
            c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
            pts = (H_UNITS @ np.array([[c, -s], [s, c]]).T) * 8 + 48
            m = np.zeros((96, 96), np.uint8)
            cv2.fillPoly(m, [np.round(pts * 8).astype(np.int32)], 1, cv2.LINE_8, 3)
            tpl.append(m.astype(bool))
        _TPL = (angs, tpl)
    return _TPL


def normals_img(h, pts, nrm):
    p0 = apply_h(h, pts)
    p1 = apply_h(h, pts + 0.05 * nrm)
    n = p1 - p0
    return p0, n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


# ----------------------------------------------------------------------------
# Binarização
# ----------------------------------------------------------------------------
def odd(v):
    v = int(round(v))
    return v + 1 if v % 2 == 0 else v


def binarizations(g):
    """Gera (nome, máscara 0/255 com tinta=255) por ordem de custo/probabilidade."""
    hh, ww = g.shape[:2]
    m = min(hh, ww)
    yield "adaptativo bloco grande", cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                           cv2.THRESH_BINARY_INV, max(31, odd(m / 6)), 7)
    yield "otsu", cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    yield "adaptativo bloco pequeno", cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                            cv2.THRESH_BINARY_INV, max(15, odd(m / 24)), 5)


# ----------------------------------------------------------------------------
# Anel
# ----------------------------------------------------------------------------
def ell_resid(cnt, ell):
    """Desvio radial médio normalizado de um contorno a uma elipse (0 = elipse perfeita)."""
    (cx, cy), (w, h), ang = ell
    a, b = max(w, 1e-6) / 2, max(h, 1e-6) / 2
    t = math.radians(ang)
    c, s = math.cos(t), math.sin(t)
    d = cnt.reshape(-1, 2).astype(np.float64) - np.array([cx, cy])
    u = d[:, 0] * c + d[:, 1] * s
    v = -d[:, 0] * s + d[:, 1] * c
    return float(np.mean(np.abs(np.sqrt((u / a) ** 2 + (v / b) ** 2) - 1.0)))


def check_ring(co, ch, prm):
    """(dict, None) se (contorno exterior, buraco) é um anel; (None, motivo) caso contrário."""
    if len(co) < 20 or len(ch) < 20:
        return None, "poucos pts"
    eo, ei = cv2.fitEllipse(co), cv2.fitEllipse(ch)
    ao, bo = max(eo[1]) / 2, min(eo[1]) / 2
    ai, bi = max(ei[1]) / 2, min(ei[1]) / 2
    if 2 * bo < prm.min_ring_px:
        return None, "pequeno"
    if bo / ao < prm.min_axis_ratio:
        return None, "achatado"
    ro, ri = ell_resid(co, eo), ell_resid(ch, ei)
    tol = 0.04 + 0.7 / max(bi, 1.0)
    if ro > tol or ri > tol:
        return None, f"nao elipse {ro:.2f}/{ri:.2f}"
    ratio = (bi * ai) / (bo * ao)
    if not 0.50 <= ratio <= 0.78:
        return None, f"razao areas {ratio:.2f}"
    dc = math.hypot(eo[0][0] - ei[0][0], eo[0][1] - ei[0][1]) / ao
    if dc > 0.12:
        return None, f"nao concentrico {dc:.2f}"
    return dict(eo=eo, ei=ei, score=1.0 - (ro + ri) / (2 * tol), ratio=ratio), None


def find_h_in_ring(ring, tops, cnts, areas):
    (cx, cy), (w, h), ang = ring["ei"]
    a, b = w / 2, h / 2
    t = math.radians(ang)
    c, s = math.cos(t), math.sin(t)
    inner_area = math.pi * a * b
    best = None
    for k in tops:
        ar = areas[k]
        if not 0.08 * inner_area <= ar <= 0.45 * inner_area:
            continue
        m = cv2.moments(cnts[k])
        if m["m00"] <= 0:
            continue
        dx, dy = m["m10"] / m["m00"] - cx, m["m01"] / m["m00"] - cy
        r = math.hypot((dx * c + dy * s) / a, (-dx * s + dy * c) / b)
        if r > 0.5:
            continue
        hull = cv2.contourArea(cv2.convexHull(cnts[k]))
        if hull <= 0 or not 0.45 <= ar / hull <= 0.78:
            continue
        if best is None or ar > areas[best]:
            best = k
    return best


# ----------------------------------------------------------------------------
# Ajuste: ângulo no frame canónico, ICP, gradiente
# ----------------------------------------------------------------------------
def angle_candidates(hmask, ei):
    """Ângulos (graus) do modelo no frame do anel, por IoU decrescente (até 3 picos distintos)."""
    (cx, cy), (w, h), ang = ei
    t = math.radians(ang)
    E = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]) @ np.diag([w / 2, h / 2])
    Ei = np.linalg.inv(E)
    A = 32.0 * Ei
    b = 48.0 - A @ np.array([cx, cy])
    canon = cv2.warpAffine(hmask, np.hstack([A, b[:, None]]), (96, 96), flags=cv2.INTER_LINEAR) > 127
    angs, tpl = templates()
    iou = np.array([(canon & m).sum() / max(1, (canon | m).sum()) for m in tpl])
    order = np.argsort(-iou)
    picks = []
    n = len(angs)
    for i in order:
        if all(min(abs(i - j), n - abs(i - j)) * 2 >= 16 for j in picks):
            picks.append(int(i))
        if len(picks) == 3:
            break
    return E, np.array([cx, cy]), [(float(angs[i]), float(iou[i])) for i in picks], canon, Ei


def hom_from_ring(E, c0, psi):
    c, s = math.cos(math.radians(psi)), math.sin(math.radians(psi))
    L = E @ np.array([[c, -s], [s, c]]) / 4.0
    h = np.eye(3)
    h[:2, :2] = L
    h[:2, 2] = c0
    return h


def icp_binary(hom, cont, iters=25):
    """ICP ponto-a-reta: arestas do modelo projetadas -> contorno binário do H."""
    n_it = 0
    for it in range(iters):
        xpx = scale_of(hom)
        gate = max(2.0, xpx * (0.8 if it < 6 else 0.35))
        P, n = normals_img(hom, MS.pts, MS.nrm)
        d2 = ((P[:, None, :] - cont[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        dd = np.sqrt(d2[np.arange(len(P)), j])
        v = cont[j] - P
        along = (v * n).sum(1)
        ok = dd <= gate
        if ok.sum() < 30:
            return hom, n_it, False
        thr = np.quantile(np.abs(along[ok]), 0.85)
        keep = ok & (np.abs(along) <= thr)
        tgt = P + n * along[:, None]
        new, _ = cv2.findHomography(MS.pts[keep], tgt[keep], 0)
        if new is None or not np.isfinite(new).all() or np.linalg.det(new[:2, :2]) <= 0:
            return hom, n_it, False
        n_it += 1
        move = float(np.mean(np.linalg.norm(apply_h(new, MS.pts[keep]) - P[keep], axis=1)))
        hom = new / new[2, 2]
        if move < 0.01:
            break
    return hom, n_it, True


def gray_refine(hom, g, iters=3):
    """Refinamento sub-pixel: máximo do gradiente ao longo da normal de cada aresta."""
    xpx = scale_of(hom)
    if xpx < 2.5:
        return hom, None
    rng_px = float(np.clip(0.35 * xpx, 1.2, 3.0))
    step = 0.25
    offs = np.arange(-rng_px, rng_px + 1e-6, step)
    sel = MS.dv >= 0.2
    pts, nrm = MS.pts[sel], MS.nrm[sel]
    info = None
    for _ in range(iters):
        P, n = normals_img(hom, pts, nrm)
        X = (P[:, 0:1] + n[:, 0:1] * offs[None, :]).astype(np.float32)
        Y = (P[:, 1:2] + n[:, 1:2] * offs[None, :]).astype(np.float32)
        prof = cv2.remap(g, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        gr = np.gradient(prof, step, axis=1)  # tinta escura => intensidade sobe para fora
        k = gr.argmax(1)
        ar = np.arange(len(k))
        gm = gr[ar, k]
        km = np.clip(k, 1, len(offs) - 2)
        gl, gc, gh = gr[ar, km - 1], gr[ar, km], gr[ar, km + 1]
        den = gl - 2 * gc + gh
        delta = np.where(np.abs(den) > 1e-9, 0.5 * (gl - gh) / den, 0.0)
        off = offs[km] + np.clip(delta, -1, 1) * step
        thr = max(2.0, 0.35 * float(np.percentile(gm, 75)))
        valid = (k > 0) & (k < len(offs) - 1) & (gm >= thr)
        if valid.sum() < 30:
            return hom, dict(support=float(valid.mean()), rms=float("nan"), P=P, E=P, valid=valid)
        thr_r = np.quantile(np.abs(off[valid]), 0.9)
        keep = valid & (np.abs(off) <= thr_r)
        tgt = P + n * off[:, None]
        new, _ = cv2.findHomography(pts[keep], tgt[keep], 0)
        if new is None or not np.isfinite(new).all() or np.linalg.det(new[:2, :2]) <= 0:
            break
        info = dict(support=float(valid.mean()), rms=float(np.sqrt(np.mean(off[keep] ** 2))), P=P, E=tgt, valid=keep)
        hom = new / new[2, 2]
    return hom, info


def polygon_iou(hom, hmask, hcnt):
    poly = apply_h(hom, H_UNITS)
    x, y, w, h = cv2.boundingRect(hcnt)
    x0 = int(max(0, min(x, poly[:, 0].min()) - 2))
    y0 = int(max(0, min(y, poly[:, 1].min()) - 2))
    x1 = int(min(hmask.shape[1], max(x + w, poly[:, 0].max()) + 3))
    y1 = int(min(hmask.shape[0], max(y + h, poly[:, 1].max()) + 3))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    pm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(pm, [np.round((poly - [x0, y0]) * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
    a, b = pm > 0, hmask[y0:y1, x0:x1] > 127
    return float((a & b).sum() / max(1, (a | b).sum()))


def fit_from_hom0(hom0, hcnt, hmask, g, prm):
    cont = hcnt.reshape(-1, 2).astype(np.float64)
    if len(cont) > 500:
        cont = cont[::int(math.ceil(len(cont) / 500))]
    hom, n_it, ok = icp_binary(hom0, cont)
    if not ok:
        return None
    iou = polygon_iou(hom, hmask, hcnt)
    hom_b = hom
    ginfo = None
    if prm.gray_refine:
        hom2, ginfo = gray_refine(hom, g)
        if polygon_iou(hom2, hmask, hcnt) >= iou - 0.05:  # só aceita se não piorar a concordância binária
            hom = hom2
        else:
            ginfo = None
    return dict(hom=hom, hom_b=hom_b, hom0=hom0, iou=iou, n_it=n_it, ginfo=ginfo, cont=cont)


def match_cyclic(poly12):
    """Correspondência cíclica entre 12 vértices da imagem e o modelo; devolve (hom, erro rms)."""
    pts = poly12.astype(np.float64)
    sm = cv2.contourArea(H_UNITS.astype(np.float32).reshape(-1, 1, 2), True)
    si = cv2.contourArea(pts.astype(np.float32).reshape(-1, 1, 2), True)
    if sm * si < 0:
        pts = pts[::-1]
    best = (None, 1e9)
    for s in range(NK):
        src = H_UNITS[(np.arange(NK) + s) % NK]
        hom, _ = cv2.findHomography(src, pts, 0)
        if hom is None or not np.isfinite(hom).all() or np.linalg.det(hom[:2, :2]) <= 0:
            continue
        e = float(np.sqrt(np.mean(np.sum((apply_h(hom, src) - pts) ** 2, axis=1))))
        if e < best[1]:
            best = (hom / hom[2, 2], e)
    return best


# ----------------------------------------------------------------------------
# Análise de uma máscara binária
# ----------------------------------------------------------------------------
def analyse_mask(mask, g, prm, dbg):
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    dbg["cnts"], dbg["hier"] = cnts, (hier[0] if hier is not None else None)
    dbg["rings"], dbg["rejects"] = [], []
    if hier is None:
        return None
    hier = hier[0]
    tops = [i for i in range(len(cnts)) if hier[i][3] == -1 and len(cnts[i]) >= 8]
    areas = {i: cv2.contourArea(cnts[i]) for i in tops}
    min_area = math.pi * (prm.min_ring_px / 2) ** 2
    ring_cands = sorted([i for i in tops if hier[i][2] != -1 and areas[i] >= min_area], key=lambda k: -areas[k])
    fits = []
    for i in ring_cands[:prm.max_ring_cand]:
        holes, c = [], hier[i][2]
        while c != -1:
            holes.append(c)
            c = hier[c][0]
        hole = max(holes, key=lambda k: cv2.contourArea(cnts[k]))
        ring, why = check_ring(cnts[i], cnts[hole], prm)
        if ring is None:
            dbg["rejects"].append((cv2.fitEllipse(cnts[i]) if len(cnts[i]) >= 5 else None, why))
            continue
        k = find_h_in_ring(ring, tops, cnts, areas)
        if k is None:
            dbg["rejects"].append((ring["eo"], "sem H dentro"))
            continue
        dbg["rings"].append(ring)
        hmask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(hmask, [cnts[k]], -1, 255, -1)
        E, c0, cands, canon, Ei = angle_candidates(hmask, ring["ei"])
        best = None
        for psi, iou0 in cands:
            r = fit_from_hom0(hom_from_ring(E, c0, psi), cnts[k], hmask, g, prm)
            if r is not None and (best is None or r["iou"] > best["iou"]):
                best = r
                best.update(psi=psi, iou0=iou0)
        if best is None:
            dbg["rejects"].append((ring["eo"], "ICP falhou"))
            continue
        best.update(mode="ring", ring=ring, hcnt=cnts[k], hmask=hmask, canon=canon, E=E, c0=c0, Ei=Ei, cands=cands)
        fits.append(best)
        if best["iou"] >= 0.92:
            break
    if fits:
        return max(fits, key=lambda f: f["iou"])

    if prm.h_only:  # modo de recurso: H sem anel
        cand = []
        for i in tops:
            if areas[i] < prm.min_h_area:
                continue
            hull = cv2.contourArea(cv2.convexHull(cnts[i]))
            if hull > 0 and 0.45 <= areas[i] / hull <= 0.78:
                cand.append(i)
        cand.sort(key=lambda k: -areas[k])
        for i in cand[:30]:
            per = cv2.arcLength(cnts[i], True)
            poly = None
            for eps in np.linspace(0.008, 0.05, 12) * per:
                ap = cv2.approxPolyDP(cnts[i], eps, True)
                if len(ap) == NK:
                    poly = ap.reshape(-1, 2)
                    break
            if poly is None:
                continue
            hom0, e = match_cyclic(poly)
            if hom0 is None or e > max(1.5, 0.15 * scale_of(hom0)):
                continue
            hmask = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(hmask, [cnts[i]], -1, 255, -1)
            r = fit_from_hom0(hom0, cnts[i], hmask, g, prm)
            if r is not None:
                r.update(mode="h_only", ring=None, hcnt=cnts[i], hmask=hmask, canon=None, cands=[], psi=None, iou0=None)
                fits.append(r)
                if r["iou"] >= 0.92:
                    break
    return max(fits, key=lambda f: f["iou"]) if fits else None


# ----------------------------------------------------------------------------
# Visualização
# ----------------------------------------------------------------------------
def to_bgr(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im.copy()


def crop_box(img, box, margin=0.35, min_size=80):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    s = max(w, h, min_size) * (1 + 2 * margin) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0, Y0 = int(max(0, cx - s)), int(max(0, cy - s))
    X1, Y1 = int(min(img.shape[1], cx + s)), int(min(img.shape[0], cy + s))
    return img[Y0:Y1, X0:X1]


def fit_tile(img, tw, th, title):
    img = to_bgr(img)
    ih, iw = img.shape[:2]
    s = min(tw / iw, th / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    tile = np.full((th, tw, 3), 28, np.uint8)
    y0, x0 = (th - r.shape[0]) // 2, (tw - r.shape[1]) // 2
    tile[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    put_text(tile, title, (6, 18), 0.5, (0, 255, 255))
    return tile


def make_mosaic(stages, tile_w, aspect, cols=3):
    th = int(round(tile_w / aspect))
    tiles = [fit_tile(im, tile_w, th, name) for name, im in stages]
    while len(tiles) % cols:
        tiles.append(np.full((th, tile_w, 3), 28, np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def draw_poly(im, pts, col, th=1):
    cv2.polylines(im, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], True, col, th, cv2.LINE_AA)


def draw_ids(im, k, col=(0, 255, 0)):
    for i, (x, y) in enumerate(k):
        cv2.circle(im, (int(round(x)), int(round(y))), 3, col, -1, cv2.LINE_AA)
        cv2.putText(im, str(i), (int(round(x)) + 4, int(round(y)) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Deteção
# ----------------------------------------------------------------------------
def detect(img, prm, debug=False):
    """Devolve dict: found, kxy (12,2), mode, iou, x_px, support, rms_edge, up_deg, flip_ambig, reason, stages, ms."""
    t0 = time.perf_counter()
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray0, (0, 0), prm.blur) if prm.blur > 0 else gray0
    pols = ["dark", "light"] if prm.polarity == "auto" else [prm.polarity]
    best, first_mask, last_dbg, tried = None, None, None, 0
    for pol in pols:
        g = gray if pol == "dark" else 255 - gray
        gf = cv2.GaussianBlur(g, (0, 0), 0.8).astype(np.float32) if prm.gray_refine else None
        for name, mask in binarizations(g):
            dbg = {}
            tried += 1
            if first_mask is None:
                first_mask = (f"{name} ({pol})", mask)
            fit = analyse_mask(mask, gf if gf is not None else g.astype(np.float32), prm, dbg)
            if last_dbg is None or fit is not None:
                last_dbg = dict(dbg, mask=mask, mask_name=f"{name} ({pol})", pol=pol)
            if fit is not None and (best is None or fit["iou"] > best["iou"]):
                best = fit
                best.update(dbg=dict(dbg, mask=mask, mask_name=f"{name} ({pol})", pol=pol))
            if best is not None and best["iou"] >= 0.9:
                break
        if best is not None and best["iou"] >= 0.9:
            break

    out = dict(found=False, kxy=None, mode=None, iou=0.0, x_px=0.0, support=float("nan"), rms_edge=float("nan"),
               up_deg=float("nan"), flip_ambig=False, reason="", stages=None)
    if best is None:
        out["reason"] = "sem anel+H" + (" nem H isolado" if prm.h_only else "")
    elif best["iou"] < prm.min_iou:
        out["reason"] = f"IoU {best['iou']:.2f} < {prm.min_iou:.2f}"
    else:
        hom = best["hom"]
        top = apply_h(hom, H_UNITS[[0, 1, 4, 5]]).mean(0)
        bot = apply_h(hom, H_UNITS[[6, 7, 10, 11]]).mean(0)
        up = top - bot
        ang = math.degrees(math.atan2(up[0], -up[1]))  # 0 = topo do H para cima na imagem; + = horário
        if abs(ang) > 90:  # hipótese rodada 180°: aplica a simetria do H
            hom = hom @ np.diag([-1.0, -1.0, 1.0])
            ang = (ang + 180) % 360
            ang = ang - 360 if ang > 180 else ang
        best["hom_final"] = hom
        ginfo = best["ginfo"]
        out.update(found=True, kxy=apply_h(hom, H_UNITS), mode=best["mode"], iou=best["iou"], x_px=scale_of(hom),
                   support=(ginfo["support"] if ginfo else float("nan")),
                   rms_edge=(ginfo["rms"] if ginfo else float("nan")), up_deg=ang, flip_ambig=abs(ang) > prm.yaw_max)
    out["ms"] = (time.perf_counter() - t0) * 1000.0
    if debug:
        out["stages"] = build_stages(img, gray, best, last_dbg, first_mask, out, prm)
    return out


def blank(img, text):
    b = np.full(img.shape[:2] if img.ndim == 2 else (img.shape[0], img.shape[1]), 28, np.uint8)
    b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    put_text(b, text, (10, img.shape[0] // 2), 0.7, (160, 160, 160))
    return b


def build_stages(img, gray, best, dbg, first_mask, out, prm):
    st = [("1 original", img), ("2 cinzento+blur (sigma %.1f)" % prm.blur, gray)]
    if dbg is None:
        dbg = {}
    mask = dbg.get("mask", first_mask[1] if first_mask else np.zeros(gray.shape, np.uint8))
    mname = dbg.get("mask_name", first_mask[0] if first_mask else "")
    st.append(("3 binaria: " + mname, mask))

    # 4: contornos e candidatos a anel
    im4 = (to_bgr(img) * 0.45).astype(np.uint8)
    cnts, hier = dbg.get("cnts", []), dbg.get("hier")
    if hier is not None:
        big = sorted(range(len(cnts)), key=lambda i: -len(cnts[i]))[:400]
        for i in big:
            cv2.drawContours(im4, [cnts[i]], -1, (255, 140, 0) if hier[i][3] == -1 else (0, 200, 255), 1)
    for ring in dbg.get("rings", []):
        cv2.ellipse(im4, ring["eo"], (0, 255, 0), 2, cv2.LINE_AA)
        cv2.ellipse(im4, ring["ei"], (0, 255, 0), 1, cv2.LINE_AA)
    for ell, why in dbg.get("rejects", []):
        if ell is not None:
            cv2.ellipse(im4, ell, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(im4, why, (int(ell[0][0]) - 30, int(ell[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 255), 1, cv2.LINE_AA)
    put_text(im4, "azul=exteriores amarelo=buracos verde=anel aceite vermelho=rejeitado", (6, im4.shape[0] - 8), 0.45)
    st.append(("4 contornos / aneis", im4))

    if best is None:
        for nm in STAGE_NAMES[4:8]:
            st.append((nm, blank(img, "n/a")))
        res = to_bgr(img)
        put_text(res, "NAO DETETADO: " + out["reason"], (10, 30), 0.7, (0, 165, 255))
        st.append(("9 resultado", res))
        return st

    cx = best.get("dbg", dbg)
    hom, hom_b, hom0 = best["hom"], best["hom_b"], best["hom0"]
    allp = np.vstack([apply_h(hom, H_UNITS), apply_h(hom0, H_UNITS), apply_h(hom_b, H_UNITS)])
    if best["ring"] is not None:
        (ex, ey), (ew, eh), _ = best["ring"]["eo"]
        rr = max(ew, eh) / 2
        box = (ex - rr, ey - rr, ex + rr, ey + rr)
    else:
        box = (allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max())

    # 5: anel + H
    im5 = to_bgr(img)
    if best["ring"] is not None:
        cv2.ellipse(im5, best["ring"]["eo"], (0, 255, 0), 1, cv2.LINE_AA)
        cv2.ellipse(im5, best["ring"]["ei"], (255, 0, 255), 1, cv2.LINE_AA)
    cv2.drawContours(im5, [best["hcnt"]], -1, (0, 0, 255), 1)
    st.append((f"5 anel + H ({best['mode']})", crop_box(im5, box)))

    # 6: frame canónico (só no modo anel)
    if best["canon"] is not None:
        warp = cv2.warpAffine(gray, np.hstack([32.0 * best["Ei"], (48.0 - 32.0 * best["Ei"] @ best["c0"])[:, None]]),
                              (96, 96), flags=cv2.INTER_LINEAR)
        im6 = cv2.resize(to_bgr(warp), (288, 288), interpolation=cv2.INTER_NEAREST)
        cm = cv2.resize(best["canon"].astype(np.uint8) * 255, (288, 288), interpolation=cv2.INTER_NEAREST)
        im6[cm > 0] = (0.55 * im6[cm > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        c, s = math.cos(math.radians(best["psi"])), math.sin(math.radians(best["psi"]))
        pts = ((H_UNITS @ np.array([[c, -s], [s, c]]).T) * 8 + 48) * 3
        draw_poly(im6, pts, (0, 0, 255), 2)
        cv2.circle(im6, (144, 144), 96, (255, 0, 255), 1, cv2.LINE_AA)
        put_text(im6, "angulos (deg:IoU): " + " ".join(f"{a:.0f}:{v:.2f}" for a, v in best["cands"]), (4, 280), 0.42)
        st.append(("6 frame canonico (anel -> circulo)", im6))
    else:
        st.append(("6 frame canonico", blank(img, "modo h_only (sem anel)")))

    # 7: ICP binário
    im7 = to_bgr(img)
    for x, y in best["cont"]:
        cv2.circle(im7, (int(x), int(y)), 0, (255, 255, 255), -1)
    draw_poly(im7, apply_h(hom0, H_UNITS), (0, 255, 255), 1)
    draw_poly(im7, apply_h(hom_b, H_UNITS), (255, 255, 0), 1)
    put_text(im7, f"amarelo=inicial ciano=ICP | {best['n_it']} it | IoU {best['iou']:.3f}", (4, im7.shape[0] - 8), 0.5)
    st.append(("7 ICP binario", crop_box(im7, box)))

    # 8: arestas em cinzento
    im8 = to_bgr(img)
    gi = best["ginfo"]
    if gi is not None and gi.get("P") is not None:
        for (px, py), (ex_, ey_), v in zip(gi["P"], gi["E"], gi["valid"]):
            col = (0, 255, 0) if v else (0, 0, 255)
            cv2.line(im8, (int(px), int(py)), (int(ex_), int(ey_)), col, 1)
        draw_poly(im8, apply_h(hom, H_UNITS), (255, 255, 0), 1)
        put_text(im8, f"suporte {gi['support'] * 100:.0f}% | rms {gi['rms']:.2f}px (verde=usado)", (4, im8.shape[0] - 8), 0.5)
    else:
        put_text(im8, "sem refinamento (H pequeno ou sem gradiente)", (4, im8.shape[0] - 8), 0.5, (0, 165, 255))
    st.append(("8 arestas (cinzento, sub-pixel)", crop_box(im8, box)))

    res = to_bgr(img)
    if out["found"]:
        draw_poly(res, out["kxy"], (255, 255, 0), 1)
        draw_ids(res, out["kxy"])
        put_text(res, f"{out['mode']} | IoU {out['iou']:.2f} | x={out['x_px']:.1f}px | {out['ms']:.0f} ms", (10, 24), 0.55)
        if out["flip_ambig"]:
            put_text(res, "AMBIGUO 180 deg (H de lado)", (10, 50), 0.55, (0, 165, 255))
    else:
        put_text(res, "NAO DETETADO: " + out["reason"], (10, 30), 0.7, (0, 165, 255))
    st.append(("9 resultado", res))
    return st


# ----------------------------------------------------------------------------
# Pose (opcional) e labels
# ----------------------------------------------------------------------------
def pose_from(out, K, x_m, prev_R=None):
    if not out["found"] or K is None:
        return None
    return estimate_pose(out["kxy"], np.ones(NK), K, obj_pts(x_m), 0.5, prev_R)


def yolo_label(out, w, h, pad=0.03):
    k = out["kxy"]
    x0, y0, x1, y1 = k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max()
    bw, bh = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - pad * bw, x1 + pad * bw, y0 - pad * bh, y1 + pad * bh
    f = [0, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
    for x, y in k:
        vis = 2 if 0 <= x < w and 0 <= y < h else 0
        f += [x / w, y / h, vis]
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in f)


def hud(vis, out, pose):
    y = 24
    if out["found"]:
        y = put_text(vis, f"H {out['mode']} | IoU {out['iou']:.2f} | x={out['x_px']:.1f}px/unid | {out['ms']:.0f} ms", (10, y))
        if out["flip_ambig"]:
            y = put_text(vis, "AMBIGUO 180 deg: H de lado", (10, y), col=(0, 165, 255))
    else:
        y = put_text(vis, "H nao detetado: " + out["reason"], (10, y), col=(0, 165, 255))
    if pose is not None:
        r, p, yw = pose["att"]
        y = put_text(vis, f"N {pose['ned'][0]:+.3f} E {pose['ned'][1]:+.3f} alt {pose['alt']:.3f} m | dist {pose['dist']:.3f} m", (10, y))
        put_text(vis, f"roll {r:+.1f} pitch {p:+.1f} yaw {yw:+.1f} deg | rms {pose['rms']:.2f}px", (10, y))


# ----------------------------------------------------------------------------
# Subcomandos
# ----------------------------------------------------------------------------
def prm_from(args):
    return default_params(polarity=args.polarity, blur=args.blur, min_ring_px=args.min_ring_px, min_iou=args.min_iou,
                          yaw_max=args.yaw_max, gray_refine=not args.no_gray_refine, h_only=not args.no_h_only)


def load_images(path):
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
    return [p]


def cmd_image(args):
    prm = prm_from(args)
    files = [f for pth in args.paths for f in load_images(pth)]
    if not files:
        sys.exit("Sem imagens.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    K = None
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"{f}: não consegui ler")
            continue
        h, w = img.shape[:2]
        if K is None or K_size != (w, h):
            ns = SimpleNamespace(calib=args.calib, hfov=args.hfov)
            K, dist, _ = prepare_K(ns, w, h)
            K_size = (w, h)
        res = detect(img, prm, debug=True)
        pose = pose_from(res, K, args.x_cm / 100.0)
        mosaic = make_mosaic(res["stages"], args.tile_w, w / h)
        cv2.imwrite(str(out_dir / f"{f.stem}_stages.png"), mosaic)
        line = f"{f.name}: " + (f"OK {res['mode']} IoU {res['iou']:.3f} x={res['x_px']:.1f}px " if res["found"]
                                else f"NAO DETETADO ({res['reason']}) ") + f"| {res['ms']:.0f} ms"
        if res["found"] and not math.isnan(res["support"]):
            line += f" | suporte arestas {res['support'] * 100:.0f}% rms {res['rms_edge']:.2f}px"
        if res["found"] and res["flip_ambig"]:
            line += " | AMBIGUO 180°"
        print(line)
        if res["found"] and len(files) == 1:
            for i, (x, y) in enumerate(res["kxy"]):
                print(f"   kp{i:2d}: {x:8.2f} {y:8.2f}")
        if pose is not None:
            r, p_, yw = pose["att"]
            print(f"   pose: N {pose['ned'][0]:+.3f} E {pose['ned'][1]:+.3f} alt {pose['alt']:.3f} m | "
                  f"roll {r:+.1f} pitch {p_:+.1f} yaw {yw:+.1f}° | rms {pose['rms']:.2f}px (K {'calibrada' if args.calib else 'provisória'})")
        if args.save_labels and res["found"]:
            (out_dir / f"{f.stem}.txt").write_text(yolo_label(res, w, h) + "\n")
        if args.show:
            cv2.imshow("classical_h (qualquer tecla = seguinte, q = sair)", mosaic)
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                break
    print(f"Mosaicos em {out_dir}/")
    if args.show:
        cv2.destroyAllWindows()


def cmd_webcam(args):
    prm = prm_from(args)
    src = Source(args)
    ok, frame = src.read()
    if not ok:
        sys.exit("Sem imagem da fonte.")
    h, w = frame.shape[:2]
    K, dist, calib_rms = prepare_K(SimpleNamespace(calib=args.calib, hfov=args.hfov), w, h)
    maps = None
    if dist is not None and np.abs(dist).max() > 1e-6:
        maps = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)
    x_m = args.x_cm / 100.0
    view = 0 if args.view == "mosaic" else int(args.view)
    paused, first, prev_R, fps_s = False, True, None, 0.0
    save_dir = Path(args.save_dir)
    log_f = log_w = None
    if args.log:
        log_f = open(args.log, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["t_s", "frame", "found", "mode", "iou", "x_px", "ms", "roll_deg", "pitch_deg", "yaw_deg",
                        "N_m", "E_m", "alt_m", "dist_m", "rms_px", "flip_ambig"])
    t_start, n, n_det = time.perf_counter(), 0, 0
    shown = None
    print("Teclas: q sai | p pausa | m mosaico | 1..9 fase em ecrã inteiro | f resultado | s guarda")
    while True:
        if not paused or shown is None:
            if first:
                first = False
            else:
                ok, frame = src.read()
                if not ok:
                    break
            clean = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR) if maps is not None else frame
            res = detect(clean, prm, debug=True)
            n += 1
            n_det += int(res["found"])
            fps_s = 1000 / res["ms"] if fps_s == 0 else 0.9 * fps_s + 0.1 * (1000 / max(res["ms"], 1e-3))
            pose = pose_from(res, K, x_m, prev_R)
            prev_R = pose["R"] if pose is not None else None
            stages = res["stages"]
            if pose is not None:
                draw_pose(stages[8][1], pose, K, x_m)
            hud(stages[8][1], res, pose)
            put_text(stages[8][1], f"FPS {fps_s:4.1f}" + ("" if calib_rms is not None else " | K PROVISORIA"),
                     (10, stages[8][1].shape[0] - 10), 0.5, (0, 165, 255) if calib_rms is None else (255, 255, 255))
            if log_w and pose is not None:
                r, p_, yw = pose["att"]
                log_w.writerow([f"{time.perf_counter() - t_start:.3f}", n, 1, res["mode"], f"{res['iou']:.3f}",
                                f"{res['x_px']:.2f}", f"{res['ms']:.1f}", f"{r:.2f}", f"{p_:.2f}", f"{yw:.2f}",
                                f"{pose['ned'][0]:.4f}", f"{pose['ned'][1]:.4f}", f"{pose['alt']:.4f}",
                                f"{pose['dist']:.4f}", f"{pose['rms']:.3f}", int(res["flip_ambig"])])
            elif log_w:
                log_w.writerow([f"{time.perf_counter() - t_start:.3f}", n, int(res["found"])] + [""] * 13)
            if args.max_frames and n >= args.max_frames:
                break
        canvas = (make_mosaic(stages, args.tile_w, w / h) if view == 0
                  else (stages[view - 1][1] if view <= 9 else stages[8][1]))
        shown = canvas
        if not args.no_display:
            cv2.imshow("classical_h", canvas)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("p"):
                paused = not paused
            elif k == ord("m"):
                view = 0
            elif k == ord("f"):
                view = 9
            elif ord("1") <= k <= ord("9"):
                view = k - ord("0")
            elif k == ord("s"):
                save_dir.mkdir(parents=True, exist_ok=True)
                name = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(save_dir / f"{name}_frame.png"), clean)
                cv2.imwrite(str(save_dir / f"{name}_stages.png"), make_mosaic(stages, args.tile_w, w / h))
                print("guardado", save_dir / f"{name}_*.png")
    src.release()
    if log_f:
        log_f.close()
    cv2.destroyAllWindows()
    print(f"{n} frames | H detetado em {100 * n_det / max(n, 1):.0f}% | {n / max(time.perf_counter() - t_start, 1e-6):.1f} fps")


def read_gt(label_path, iw, ih):
    lines = [l for l in Path(label_path).read_text().splitlines() if l.strip()] if Path(label_path).exists() else []
    if not lines:
        return None
    v = np.array(lines[0].split(), dtype=float)
    kp = v[5:].reshape(-1, 3).copy()
    kp[:, 0] *= iw
    kp[:, 1] *= ih
    return kp


def cmd_test(args):
    import json
    ds = Path(args.data)
    prm = prm_from(args)
    meta_p = ds / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else None
    K = np.array(meta["K"], dtype=np.float64) if meta else None
    P = np.array(meta["vertices_3d_m"], dtype=np.float64) if meta else None
    files = sorted(f for f in (ds / "images" / args.split).iterdir() if f.suffix.lower() in IMG_EXT)
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"Sem imagens em {ds / 'images' / args.split}")
    rows, neg_total, neg_fp, times = [], 0, 0, []
    fails = []
    for f in files:
        img = cv2.imread(str(f))
        ih, iw = img.shape[:2]
        gt = read_gt(ds / "labels" / args.split / f"{f.stem}.txt", iw, ih)
        res = detect(img, prm)
        times.append(res["ms"])
        if gt is None:
            neg_total += 1
            if res["found"]:
                neg_fp += 1
                fails.append((f, "falso positivo"))
            continue
        vis = gt[:, 2] > 0
        xg = 0.5 * (np.linalg.norm(gt[5, :2] - gt[0, :2]) / 4 + np.linalg.norm(gt[11, :2] - gt[0, :2]) / 5)
        row = dict(file=f.name, x_gt=xg, found=res["found"], err=float("nan"), err_x=float("nan"), flip=False,
                   rot=float("nan"), tr=float("nan"), dist=float("nan"))
        pj = ds / "poses" / args.split / f"{f.stem}.json"
        pose_js = json.loads(pj.read_text()) if pj.exists() else None
        if pose_js:
            row["dist"] = pose_js["distance_m"]
        if res["found"]:
            k = res["kxy"]
            e = np.linalg.norm(k[vis] - gt[vis, :2], axis=1)
            e180 = np.linalg.norm(k[(np.arange(NK) + 6) % NK][vis] - gt[vis, :2], axis=1)
            row["flip"] = bool(e180.mean() < e.mean())
            row["err"], row["err_x"] = float(e.mean()), float(e.mean() / max(xg, 1e-6))
            if K is not None and pose_js and not row["flip"] and row["err_x"] < 0.5:
                s = solve_pose(k, K, None, P)
                if s is not None:
                    rv, tv = s
                    row["rot"] = rot_err_deg(cv2.Rodrigues(rv)[0], np.array(pose_js["R"]))
                    row["tr"] = float(np.linalg.norm(tv - np.array(pose_js["t_m"])) * 1000)
            if row["flip"] or row["err_x"] > 0.5:
                fails.append((f, "flip 180°" if row["flip"] else f"erro {row['err']:.1f}px"))
        else:
            fails.append((f, "nao detetado: " + res["reason"]))
        rows.append(row)

    def stats(sel):
        n = len(sel)
        det = [r for r in sel if r["found"]]
        good = [r for r in det if not r["flip"] and r["err_x"] <= 0.5]
        errs = np.array([r["err"] for r in good]) if good else np.array([np.nan])
        ex = np.array([r["err_x"] for r in good]) if good else np.array([np.nan])
        return (n, 100 * len(det) / max(n, 1), 100 * len(good) / max(n, 1), sum(r["flip"] for r in det),
                np.nanmedian(errs), np.nanpercentile(errs, 95), np.nanmedian(ex))

    print(f"\n{len(files)} imagens ({len(rows)} com H, {neg_total} negativas) | {np.mean(times):.1f} ms/img "
          f"(p95 {np.percentile(times, 95):.1f} ms)")
    print(f"falsos positivos nas negativas: {neg_fp}/{neg_total}")
    print("\nPor tamanho do H na imagem (x = px por unidade; H tem 4x de largura):")
    print("   x (px)     |   n | detetado% | correto% | flips | erro med px | erro p95 px | erro med (x)")
    edges = [0, 3, 5, 8, 12, 20, 40, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r["x_gt"] < hi]
        if sel:
            n, d, g, fl, m, p95, mx = stats(sel)
            print(f"  {lo:4.0f}..{min(hi, 999):4.0f}   | {n:3d} | {d:9.0f} | {g:8.0f} | {fl:5d} | {m:11.2f} | {p95:11.2f} | {mx:12.3f}")
    n, d, g, fl, m, p95, mx = stats(rows)
    print(f"  TOTAL       | {n:3d} | {d:9.0f} | {g:8.0f} | {fl:5d} | {m:11.2f} | {p95:11.2f} | {mx:12.3f}")
    rot = np.array([r["rot"] for r in rows if not math.isnan(r["rot"])])
    tr = np.array([r["tr"] for r in rows if not math.isnan(r["tr"])])
    if len(rot):
        print(f"\nPose 3D (PnP sobre os keypoints clássicos, {len(rot)} imagens): rotação mediana {np.median(rot):.2f}° "
              f"(p95 {np.percentile(rot, 95):.2f}°) | translação mediana {np.median(tr):.1f} mm (p95 {np.percentile(tr, 95):.1f} mm)")
    if args.save_fail and fails:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for f, why in fails[:args.save_fail]:
            img = cv2.imread(str(f))
            res = detect(img, prm, debug=True)
            cv2.imwrite(str(out / f"fail_{f.stem}.png"), make_mosaic(res["stages"], args.tile_w, img.shape[1] / img.shape[0]))
            print(f"  falha: {f.name} ({why}) -> {out}/fail_{f.stem}.png")


def add_det_args(p):
    p.add_argument("--polarity", choices=["dark", "light", "auto"], default="dark", help="tinta escura em fundo claro (dark) ou o inverso")
    p.add_argument("--blur", type=float, default=1.0, help="sigma do desfoque gaussiano")
    p.add_argument("--min_ring_px", type=float, default=24, help="eixo menor mínimo do anel (px)")
    p.add_argument("--min_iou", type=float, default=0.80, help="IoU mínimo para aceitar a deteção")
    p.add_argument("--yaw_max", type=float, default=80.0, help="acima disto o H está 'de lado' e a ambiguidade de 180° é marcada")
    p.add_argument("--no_gray_refine", action="store_true", help="desliga o refinamento sub-pixel")
    p.add_argument("--no_h_only", action="store_true", help="desliga o modo de recurso sem anel")
    p.add_argument("--tile_w", type=int, default=440, help="largura de cada tile do mosaico")


def add_k_args(p):
    p.add_argument("--calib", default=None, help="calib.npz do webcam_test.py calibrate")
    p.add_argument("--hfov", type=float, default=70.0, help="FOV horizontal para K provisória (sem --calib)")
    p.add_argument("--x_cm", type=float, default=1.65, help="largura x do segmento do H (cm)")


def main():
    ap = argparse.ArgumentParser(description="Deteção clássica dos 12 vértices do H (anel -> H -> homografia)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("image", help="imagem(ns)/pasta: grava o mosaico com todas as fases")
    pi.add_argument("paths", nargs="+")
    pi.add_argument("--out_dir", default="classical_out")
    pi.add_argument("--show", action="store_true")
    pi.add_argument("--save_labels", action="store_true", help="grava labels YOLO-pose (pré-anotação) junto dos mosaicos")
    add_det_args(pi)
    add_k_args(pi)

    pw = sub.add_parser("webcam", help="tempo real (webcam, vídeo ou pasta) com fases do pipeline")
    add_cam_args(pw)
    add_det_args(pw)
    add_k_args(pw)
    pw.add_argument("--view", default="mosaic", help="mosaic ou 1..9 (fase em ecrã inteiro; 9 = resultado)")
    pw.add_argument("--save_dir", default="classical_captures")
    pw.add_argument("--log", default=None, help="CSV com a pose por frame")
    pw.add_argument("--max_frames", type=int, default=0)
    pw.add_argument("--no_display", action="store_true")

    pt = sub.add_parser("test", help="mede o erro contra o dataset sintético")
    pt.add_argument("--data", default="synth")
    pt.add_argument("--split", default="val")
    pt.add_argument("--limit", type=int, default=0)
    pt.add_argument("--save_fail", type=int, default=0, help="grava mosaicos das N primeiras falhas")
    pt.add_argument("--out_dir", default="classical_fail")
    add_det_args(pt)

    args = ap.parse_args()
    {"image": cmd_image, "webcam": cmd_webcam, "test": cmd_test}[args.cmd](args)


if __name__ == "__main__":
    main()
