#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_synthetic.py — Dataset sintético YOLO-pose para o heliponto "H num círculo" (OpenCV + NumPy)
================================================================================================
Edita os SETTINGS abaixo e corre o ficheiro diretamente (python gen_synthetic.py, ou Run no IDE).
Gera imagens, labels YOLO-pose (12 vértices do H; caixa = ANEL, não o H) e a pose real em JSON
(posição NED + atitude do drone + R,t do OpenCV). No fim verifica anotações e pose (MODE="generate").

Alvo: unidade x = X_CM. H com 4x de largura e 5x de altura, segmentos com espessura x; anel com raio
interior 4x e exterior 5x. Vértices em unidades x (origem no centro, X direita, Y baixo), ordem do treino:
    0(-2,-2.5) 1(-1,-2.5) 2(-1,-.5) 3(1,-.5) 4(1,-2.5) 5(2,-2.5) 6(2,2.5) 7(1,2.5) 8(1,.5) 9(-1,.5) 10(-1,2.5) 11(-2,2.5)
Referenciais: heliponto NED (N = cima do H, E = direita, D = chão); corpo FRD; câmara a nadir com o topo da
imagem = frente do drone. Atitude 3-2-1: R_ned<-corpo = Rz(yaw) Ry(pitch) Rx(roll). Pose do OpenCV:
X_cam = R @ X_alvo + t, alvo T = (X=E, Y=-N, Z=D), R = M Rnb^T M^T.
Notas: o H é igual rodado 180°, por isso |yaw| <= YAW_MAX (<90°). Sem distorção de lente (pinhole).
Sintético só para treino; val/test devem ser imagens reais. O .yaml não é gerado aqui.
"""
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# =============================== SETTINGS ====================================
MODE = "generate"                 # "generate" = gera + verifica | "verify" = só verifica OUT_DIR
OUT_DIR, SPLIT = Path(__file__).resolve().parent.parent / "dataset_v3", "train" # guardar no diretório anterior ao script
N_IMAGES, SESSIONS, SEED = 7000, 35, 0
BACKGROUNDS = Path(__file__).resolve().parent / "backgrounds"    # pasta com fotos (subpastas ok: relva/, asfalto/, interior/...), SEM helipad nem H
BG_PHOTO_PROB = 0.8               # com fotos, fração de imagens que as usa
BG_KINDS = dict(noise=0.35, grass=0.25, asphalt=0.25, text=0.15)   # fundos procedurais (pesos)
DEBUG_N = 20                      # nº de overlays (vértices, caixa, eixos N/E/D, ângulos) em OUT_DIR/debug

WIDTH, HEIGHT, HFOV = 1280, 720, 70.0   # câmara (K provisória pelo FOV horizontal)
CALIB = None                      # ou "calib.npz" (K, dist, size) em vez de HFOV
X_CM = 1.65                       # unidade x do alvo (cm)

ROLL_MAX, PITCH_MAX, YAW_MAX = 35.0, 35.0, 80.0   # graus (amostragem Latin hypercube, uniforme)
TILT_CAP = 65.0                   # inclinação total máxima da câmara
DMIN, DMAX, D_LOG = 0.15, 6.5, True   # distância ao centro do H (m); D_LOG: uniforme em log (mais perto)
MIN_BOX_PX = 25.0                 # lado mínimo da caixa do anel (px)

NEG_FRAC = 0.12                   # imagens sem H (negativos)
HARD_NEG_FRAC = 0.0               # dessas, fração com falso alvo (H sem anel, H deformado)
TEXT_NEG_FRAC = 0.1              # das restantes, fração com texto (palavras com H, letras grandes); o resto: fundo + formas
PARTIAL_FRAC = 0.12               # H parcialmente fora da imagem
OCC_PROB = 0.15                   # oclusão de um vértice (v=1)
BLUR_EXPO_MAX = 1.0               # com blur: exposição máx. do branco (1.0 = papel nunca satura; >1 recorta as rampas do blur e distorce o H; None desativa)


BLUR_PROB = 0.65                   # probabilidade de motion blur
BLUR_MAX_X = 2.5                  # comprimento máximo do blur em unidades x (distribuição ~u², mais peso aos baixos)
BLUR_MAX_PX = 90.0                # tecto absoluto do blur em px (perto, 1.5x seriam 150 px; ajusta com frames reais)
BLUR_TARGET_ONLY = 0.6            # fração em que só o alvo se mexe (câmara parada); senão a imagem toda
BLUR_V0_ABOVE_X = None            # blur > isto => keypoints do H com v=0 (None = anotar sempre os 12); quero None!!!

SHAPES = ["square", "rect", "circle", "hexagon", "octagon", "rounded", "ellipse", "blob", "quad"]
INK_RANGE = (10, 60)              # nível do preto (0-255)
PAPER_RANGE = (190, 250)          # nível do branco
DIRT = 0.5                        # sujidade/terra no pad: manchas, pintas, grão (0 = desliga)
NOISE_MAX = 0.03                  # ruído do sensor (fração de 255); sobe com pouca luz
LIGHT_GAIN = (0.8, 2.4)           # exposição por sessão (log-uniforme): <1 escuro, >1 sobreexposto
LIGHT_GAIN_IMG = 0.25             # variação extra por imagem (desvio-padrão em log)
LIGHT_GRADIENT = 0.6              # gradiente de iluminação máximo (0 = desliga)
LIGHT_SHADOW_PROB = 0.4           # probabilidade de sombra projetada
GAMMA = (0.6, 1.6)                # curva de tons
COLOR_CAST = 0.12                 # desvio máximo de cor por canal (luz quente/fria)
# ==============================================================================

NUM_KPTS = 12
H_UNITS = np.array([[-2, -2.5], [-1, -2.5], [-1, -.5], [1, -.5], [1, -2.5], [2, -2.5],
                    [2, 2.5], [1, 2.5], [1, .5], [-1, .5], [-1, 2.5], [-2, 2.5]], dtype=np.float64)
M = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
LETTERS = "ABCDEFGIJKLMNOPQRSTUVWXYZ"   # sem H (texto/confusers)
FONT = cv2.FONT_HERSHEY_SIMPLEX
TOL_REPROJ_PX, TOL_ROT_DEG, AMBIG_DEG = 0.05, 0.1, 5.0


# --------------------------- geometria e pose (API usada por outros scripts) ----------------------------
def obj_pts(x_m):
    """Vértices 3D do H no referencial do alvo (m, Z=0), shape (12,3)."""
    return np.hstack([H_UNITS * x_m, np.zeros((NUM_KPTS, 1))])


def ring_pts(x_m, n=90):
    """Pontos do contorno exterior do anel (raio 5x), para a caixa."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([5 * x_m * np.cos(a), 5 * x_m * np.sin(a), np.zeros(n)], 1)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def att_to_R(roll, pitch, yaw):
    """Atitude do drone (graus, 3-2-1) -> R do OpenCV (alvo -> câmara)."""
    return M @ (rot_z(math.radians(yaw)) @ rot_y(math.radians(pitch)) @ rot_x(math.radians(roll))).T @ M.T


def R_to_att(R):
    Rnb = M.T @ R.T @ M
    return {"roll": math.degrees(math.atan2(Rnb[2, 1], Rnb[2, 2])),
            "pitch": math.degrees(-math.asin(max(-1.0, min(1.0, Rnb[2, 0])))),
            "yaw": math.degrees(math.atan2(Rnb[1, 0], Rnb[0, 0]))}


def pose_to_ned(R, t):
    """Posição do drone (N, E, D) no referencial do heliponto."""
    return M.T @ (-R.T @ t)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def rot_err_deg(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra @ Rb.T) - 1.0) / 2.0))))


def project(P, rvec, t, K):
    return cv2.projectPoints(P, rvec, t, K, None)[0].reshape(-1, 2)


def solve_pose(img_pts, K, dist, obj):
    """PnP plano: IPPE (escolhe a melhor das 2 soluções) + refinamento LM. Devolve (rvec, t) ou None."""
    ip = img_pts.reshape(-1, 1, 2).astype(np.float64)
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, ip, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok or len(rvecs) == 0:
        return None
    i = int(np.argmin(np.asarray(errs).ravel()))
    rvec, tvec = cv2.solvePnPRefineLM(obj, ip, K, dist, rvecs[i], tvecs[i])
    return rvec, tvec.ravel()


def load_cam():
    if CALIB:
        d = np.load(CALIB)
        iw, ih = (int(v) for v in d["size"]) if "size" in d.files else (WIDTH, HEIGHT)
        return d["K"].astype(np.float64), iw, ih
    f = (WIDTH / 2.0) / math.tan(math.radians(HFOV) / 2.0)
    return np.array([[f, 0, WIDTH / 2.0], [0, f, HEIGHT / 2.0], [0, 0, 1.0]]), WIDTH, HEIGHT


# --------------------------------- pad: forma, textura, sujidade ----------------------------------------
def sheet_polygon(rng, shape):
    """Contorno do pad em unidades x, centrado no anel; raio inscrito 5.3x..9x (contém sempre o anel)."""
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    details = {}
    if shape == "square":
        p = sq
    elif shape == "rect":
        aspect = float(rng.uniform(1.2, 1.9))
        details["aspect"] = round(aspect, 3)
        p = sq * [aspect, 1]
    elif shape == "circle":
        p = np.c_[np.cos(th), np.sin(th)]
    elif shape == "ellipse":
        aspect = float(rng.uniform(1.2, 1.6))
        details["aspect"] = round(aspect, 3)
        p = np.c_[aspect * np.cos(th), np.sin(th)]
    elif shape in ("hexagon", "octagon"):
        a = np.arange(6 if shape == "hexagon" else 8) * 2 * np.pi / (6 if shape == "hexagon" else 8)
        p = np.c_[np.cos(a), np.sin(a)]
    elif shape == "rounded":
        k = rng.uniform(3.5, 7.0)
        details["corner_exponent"] = round(float(k), 3)
        p = np.c_[np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2 / k), np.sign(np.sin(th)) * np.abs(np.sin(th)) ** (2 / k)]
    elif shape == "blob":
        ph = rng.uniform(0, 2 * np.pi, 3)
        details["phase_deg"] = np.round(np.degrees(ph), 1).tolist()
        r = 1 + .12 * np.sin(3 * th + ph[0]) + .08 * np.sin(5 * th + ph[1]) + .05 * np.sin(8 * th + ph[2])
        p = np.c_[r * np.cos(th), r * np.sin(th)]
    else:  # quad irregular
        angle_offset = rng.uniform(-.35, .35, 4)
        radius = rng.uniform(1, 1.5, 4)
        details.update(angle_offset_deg=np.round(np.degrees(angle_offset), 1).tolist(),
                       vertex_radius=np.round(radius, 3).tolist())
        a = np.arange(4) * np.pi / 2 + angle_offset
        r = radius
        p = np.c_[r * np.cos(a), r * np.sin(a)]
    ang = rng.uniform(0, 2 * np.pi)
    p = p @ np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]).T
    e0, d = p, np.roll(p, -1, 0) - p
    inr = (np.abs(e0[:, 0] * d[:, 1] - e0[:, 1] * d[:, 0]) / np.linalg.norm(d, axis=1)).min()
    outer_radius = float(rng.uniform(5.3, 9.0))
    if shape not in ("circle",):
        details["rotation_deg"] = round(math.degrees(ang), 1)
    details["outer_radius_x"] = round(outer_radius, 2)
    return p * (outer_radius / inr), details


def draw_target_texture(S, ppu, paper, ink, ring=True, hs=(1.0, 1.0)):
    """Folha [-S,S]^2 (unidades x), ppu = px por unidade x. Anel 4x..5x e H (ou H deformado)."""
    n = int(round(2 * S * ppu))
    tex = np.full((n, n, 3), paper, np.uint8)
    c, sh = n / 2.0, 4
    k = 1 << sh
    if ring:
        cv2.circle(tex, (int(round(c * k)), int(round(c * k))), int(round(4.5 * ppu * k)), ink,
                   max(1, int(round(ppu))), cv2.LINE_AA, sh)
    cv2.fillPoly(tex, [np.round((H_UNITS * np.array(hs) * ppu + c) * k).astype(np.int32)], ink, cv2.LINE_AA, sh)
    return tex


def noise_field(rng, iw, ih):
    acc = np.zeros((ih, iw), np.float32)
    for cells, w in ((3, 1.0), (6, 0.6), (12, 0.35), (24, 0.2), (48, 0.1)):
        acc += w * cv2.resize(rng.random((max(2, int(cells * ih / iw)), cells)).astype(np.float32), (iw, ih),
                              interpolation=cv2.INTER_CUBIC)
    acc -= acc.min()
    return acc / (acc.max() + 1e-6)


def add_dirt(tex, ppu, rng):
    """Sujidade/terra: manchas suaves, grão e pintas castanhas (tudo ligeiro, escala com DIRT)."""
    n = tex.shape[0]
    f = tex.astype(np.float32) * (1 - 0.18 * DIRT * noise_field(rng, n, n))[..., None]
    f += rng.normal(0, 4 * DIRT, (n, n))[..., None]
    dirt = np.zeros((n, n), np.float32)
    for _ in range(int(rng.integers(0, int(80 * DIRT) + 1))):
        cv2.circle(dirt, (int(rng.uniform(0, n)), int(rng.uniform(0, n))), max(1, int(rng.uniform(.04, .22) * ppu)),
                   float(rng.uniform(.3, .9)), -1, cv2.LINE_AA)
    dirt = cv2.GaussianBlur(dirt, (0, 0), max(0.5, 0.05 * ppu))
    f = f * (1 - dirt[..., None]) + np.array([40, 60, 80]) * rng.uniform(.6, 1.3) * dirt[..., None]
    return np.clip(f, 0, 255).astype(np.uint8)


def motion_kernel(L, theta_deg):
    """Kernel linear de comprimento L px (sub-pixel por supersampling); None se L < 1."""
    if L < 1.0:
        return None
    n, ss = int(2 * math.ceil(L / 2) + 3), 8
    k = np.zeros((n * ss, n * ss), np.float32)
    c = n * ss / 2.0
    dx, dy = math.cos(math.radians(theta_deg)) * L / 2 * ss, math.sin(math.radians(theta_deg)) * L / 2 * ss
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, ss)
    k = cv2.resize(k, (n, n), interpolation=cv2.INTER_AREA)
    return k / k.sum()


def render_target(bg, pose, K, iw, ih, x_m, rng, kernel=None, fake=None, settings=None):
    """Compõe o pad sobre o fundo por homografia. kernel: motion blur só do alvo (câmara parada).
    fake: None | 'noring' | 'wrongH' (falsos alvos para negativos). Devolve (imagem, px por unidade x)."""
    poly = pose["poly"]
    upx = K[0, 0] * x_m / pose["d"]
    ppu = float(np.clip(1.5 * upx, 6.0, 60.0))
    p = int(rng.integers(*PAPER_RANGE))
    paper = tuple(int(np.clip(p * t + rng.integers(-6, 7), 0, 255)) for t in rng.uniform(0.94, 1.04, 3))
    q = int(rng.integers(*INK_RANGE))
    ink = tuple(int(np.clip(q + rng.integers(-8, 9), 0, 255)) for _ in range(3))
    if settings is not None:
        settings.update(paper_bgr=paper, ink_bgr=ink, fake=fake)
    S = float(np.abs(poly).max() + 0.2)
    kw = {}
    if fake == "noring":
        kw["ring"] = False
    elif fake == "wrongH":
        kw["hs"] = tuple(float(rng.choice([rng.uniform(.3, .6), rng.uniform(1.45, 1.8)])) for _ in range(2))
        if settings is not None:
            settings["h_scale"] = [round(v, 3) for v in kw["hs"]]
    tex = draw_target_texture(S, ppu, paper, ink, **kw)
    if DIRT > 0:
        tex = add_dirt(tex, ppu, rng)
    n = tex.shape[0]
    c, sh = n / 2.0, 4
    mask = np.zeros((n, n), np.uint8)
    cv2.fillPoly(mask, [np.round((poly * ppu + c) * (1 << sh)).astype(np.int32)], 255, cv2.LINE_AA, sh)
    s = x_m / ppu
    R, t = pose["R"], pose["t"]
    Mh = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ np.array([[s, 0, -c * s], [0, s, -c * s], [0, 0, 1.0]])
    warped = cv2.warpPerspective(tex, Mh, (iw, ih), flags=cv2.INTER_LINEAR).astype(np.float32)
    alpha = cv2.warpPerspective(mask, Mh, (iw, ih), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
    pm = warped * alpha
    if kernel is not None:  # alvo em movimento: esborrata alvo e máscara (pré-multiplicados), fundo nítido
        pm, alpha = cv2.filter2D(pm, -1, kernel), cv2.filter2D(alpha[..., 0], -1, kernel)[..., None]
    return np.clip(bg.astype(np.float32) * (1 - alpha) + pm, 0, 255).astype(np.uint8), upx


# ------------------------------------ amostragem de poses ----------------------------------------------
def sample_pose(rng, K, iw, ih, x_m, strat=None, partial=False):
    """Pose válida: todos os 12 vértices na imagem (>=6 se partial) e caixa do anel >= MIN_BOX_PX.
    strat = (i_roll, i_pitch, i_yaw, i_dist, n): faixas do Latin hypercube (relaxa após 150 tentativas)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P, RP = obj_pts(x_m), ring_pts(x_m)
    lo, hi = (-0.10, 1.10) if partial else (0.08, 0.92)
    need = 6 if partial else NUM_KPTS
    for k in range(400):
        use = strat if (strat is not None and k < 150) else None
        u01 = (lambda i: rng.random()) if use is None else (lambda i: (use[i] + rng.random()) / use[4])
        roll, pitch, yaw = (2 * u01(0) - 1) * ROLL_MAX, (2 * u01(1) - 1) * PITCH_MAX, (2 * u01(2) - 1) * YAW_MAX
        sd = u01(3)
        d = math.exp(math.log(DMIN) + sd * (math.log(DMAX) - math.log(DMIN))) if D_LOG else DMIN + sd * (DMAX - DMIN)
        R = att_to_R(roll, pitch, yaw)
        if R[2, 2] < math.cos(math.radians(TILT_CAP)):
            continue
        u, v = rng.uniform(lo * iw, hi * iw), rng.uniform(lo * ih, hi * ih)
        t = np.array([(u - cx) * d / fx, (v - cy) * d / fy, d])
        shape = str(rng.choice(SHAPES))
        poly, shape_details = sheet_polygon(rng, shape)
        corners = np.c_[poly * x_m, np.zeros(len(poly))]
        if ((R @ corners.T)[2] + d).min() < 0.03:  # pad atrás/muito perto da câmara
            continue
        rvec = cv2.Rodrigues(R)[0]
        pts, ring = project(P, rvec, t, K), project(RP, rvec, t, K)
        inside = (pts[:, 0] >= 0) & (pts[:, 0] <= iw - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= ih - 1)
        if int(inside.sum()) < need or max(np.ptp(ring[:, 0]), np.ptp(ring[:, 1])) < MIN_BOX_PX:
            continue
        return dict(R=R, rvec=rvec, t=t, pts=pts, ring=ring, inside=inside, d=d, poly=poly, shape=shape,
                    attitude=R_to_att(R), pos_ned=pose_to_ned(R, t),
                    tilt=math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))),
                    relaxed=bool(strat is not None and use is None), partial_requested=partial,
                    pad_size_units=np.ptp(poly, axis=0).tolist(), shape_details=shape_details)
    return None


# ------------------------------- fundos, confusores, oclusão, fotometria --------------------------------
BASES = [(60, 110, 70), (70, 120, 90), (110, 110, 110), (150, 150, 150), (90, 95, 100),
         (100, 130, 160), (120, 140, 170), (70, 80, 90), (150, 170, 180), (80, 100, 80)]


def hsv2bgr(h, s_, v):
    return cv2.cvtColor(np.dstack([np.clip(h, 0, 179), np.clip(s_, 0, 255), np.clip(v, 0, 255)]).astype(np.uint8),
                        cv2.COLOR_HSV2BGR).reshape(-1, 3)


def noise_bg(rng, iw, ih, pxmm=1.0):
    base = np.array(BASES[int(rng.integers(len(BASES)))], np.float32) + rng.normal(0, 12, 3)
    n1, n2 = noise_field(rng, iw, ih), noise_field(rng, iw, ih)
    img = base[None, None, :] * (0.55 + 0.9 * n1[..., None]) * (0.9 + 0.2 * n2[..., None] * rng.uniform(.3, 1, 3))
    img = np.clip(img + rng.normal(0, rng.uniform(2, 10), img.shape), 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(0, 8))):  # fendas / juntas
        g = int(rng.integers(0, 255))
        cv2.line(img, (int(rng.uniform(0, iw)), int(rng.uniform(0, ih))), (int(rng.uniform(0, iw)), int(rng.uniform(0, ih))),
                 (g, g, g), int(rng.integers(1, 5)), cv2.LINE_AA)
    return img


def grass_bg(rng, iw, ih, pxmm):
    """Relva: campo verde + folhas (comprimento ~25-60 mm, largura ~2-4 mm, em px pela escala pxmm = px/mm)."""
    n1, n2 = noise_field(rng, iw, ih), noise_field(rng, iw, ih)
    img = cv2.cvtColor(np.dstack([38 + 14 * n1, 90 + 60 * n2, 25 + 45 * n1]).astype(np.uint8), cv2.COLOR_HSV2BGR)
    L, W = np.clip(rng.uniform(25, 60) * pxmm, 5, 70), np.clip(rng.uniform(2, 4) * pxmm, 1, 7)
    n = int(min(30000, 4 * iw * ih / (L * W + 1)))
    for lo, hi in ((25, 85), (55, 130), (85, 190)):  # 3 camadas, escura -> clara
        m = n // 3
        p0, ang, ln = rng.uniform(0, [iw, ih], (m, 2)), rng.uniform(0, 2 * np.pi, m), L * rng.uniform(.5, 1.2, m)
        dry = rng.random(m) < 0.05
        col = hsv2bgr(np.where(dry, rng.uniform(22, 32, m), rng.normal(46, 6, m)), rng.uniform(80, 185, m), rng.uniform(lo, hi, m))
        p1 = p0 + np.stack([np.cos(ang), np.sin(ang)], 1) * ln[:, None]
        for a_, b_, c_ in zip(p0.astype(int), p1.astype(int), col):
            cv2.line(img, tuple(a_), tuple(b_), (int(c_[0]), int(c_[1]), int(c_[2])), int(max(1, W * rng.uniform(.7, 1.3))), cv2.LINE_AA)
    return img


def asphalt_bg(rng, iw, ih, pxmm):
    """Asfalto: grão (agregado ~2-7 mm), ruído, fendas e, às vezes, uma linha pintada gasta."""
    k = max(1, int(round(rng.uniform(2, 7) * pxmm)))
    agg = cv2.resize(rng.normal(0, 16, (ih // k + 2, iw // k + 2)).astype(np.float32), (iw, ih), interpolation=cv2.INTER_CUBIC)
    f = rng.uniform(55, 125) * (0.85 + 0.3 * noise_field(rng, iw, ih)) + agg + rng.normal(0, 6, (ih, iw))
    img = np.clip(f[..., None] * rng.uniform(0.92, 1.08, 3), 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(0, 4))):  # fendas
        pts = np.cumsum(rng.normal(0, 25, (8, 2)), 0) + rng.uniform(0, [iw, ih])
        cv2.polylines(img, [pts.astype(np.int32)], False, (25, 25, 25), int(rng.integers(1, 4)), cv2.LINE_AA)
    if rng.random() < 0.4:  # linha pintada (branca ou amarela), com desgaste
        w, a, c = np.clip(rng.uniform(80, 150) * pxmm, 6, 200), rng.uniform(0, np.pi), rng.uniform(0, [iw, ih])
        d, n_ = np.array([math.cos(a), math.sin(a)]) * (iw + ih), np.array([-math.sin(a), math.cos(a)]) * w / 2
        m = np.zeros((ih, iw), np.float32)
        cv2.fillPoly(m, [np.array([c - d - n_, c + d - n_, c + d + n_, c - d + n_], np.int32)], 1.0, cv2.LINE_AA)
        m = (m * (0.55 + 0.45 * noise_field(rng, iw, ih)))[..., None]
        col = np.array([225, 225, 225] if rng.random() < .6 else [40, 190, 230], np.float32)
        img = (img * (1 - m) + col * m).astype(np.uint8)
    return img


WORDS = ("HELIPAD HOME HOUSE HOTEL HALL HANGAR HIGH ABANICO HELLO HEIGHT EXIT PARKING OFFICE DATA SPEED WARNING H HH "
         "12 305 A4 NORTH HEAT HOLD HELP").split()


def text_bg(rng, iw, ih, pxmm):
    """Página/placa com texto (muitas palavras com H) e, às vezes, um H grande isolado, sem círculo."""
    light = rng.random() < 0.8
    bg_c, fg_c = (rng.integers(170, 250), rng.integers(0, 70)) if light else (rng.integers(10, 80), rng.integers(180, 255))
    img = np.clip(bg_c + rng.normal(0, 4, (ih, iw, 3)), 0, 255).astype(np.uint8)
    fg = (int(fg_c),) * 3
    font = [FONT, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX][int(rng.integers(4))]
    th1 = cv2.getTextSize("H", font, 1.0, 1)[0][1]
    hpx = float(np.clip(rng.uniform(4, 40) * pxmm, 8, 160))
    sc, thick = hpx / th1, max(1, int(round(hpx / th1 * rng.uniform(1, 2))))
    y = int(rng.uniform(0, hpx))
    while y < ih + hpx:
        x = int(rng.uniform(-2 * hpx, hpx))
        while x < iw:
            w = str(rng.choice(WORDS))
            cv2.putText(img, w, (x, y), font, sc, fg, thick, cv2.LINE_AA)
            x += int(cv2.getTextSize(w, font, sc, thick)[0][0] + 0.6 * hpx)
        y += int(hpx * rng.uniform(1.4, 2.0))
    if rng.random() < 0.5:  # H grande de letreiro
        big = float(np.clip(rng.uniform(30, 150) * pxmm, 40, 0.6 * ih)) / th1
        cv2.putText(img, "H", (int(rng.uniform(0, iw * .8)), int(rng.uniform(ih * .3, ih))), font, big, fg,
                    max(2, int(big * 1.5)), cv2.LINE_AA)
    R = cv2.getRotationMatrix2D((iw / 2, ih / 2), float(rng.uniform(-20, 20)), 1.0)
    return cv2.warpAffine(img, R, (iw, ih), borderMode=cv2.BORDER_REFLECT)


def make_bg(rng, files, iw, ih, pxmm, kind=None, settings=None):
    """Fundo: foto (se houver) ou procedural (ruído, relva, asfalto, texto); pxmm = px por mm à distância do alvo."""
    if kind is None and files and rng.random() < BG_PHOTO_PROB:
        return photo_bg(rng, files, iw, ih, settings)
    kind = kind or str(rng.choice(list(BG_KINDS), p=np.array(list(BG_KINDS.values())) / sum(BG_KINDS.values())))
    if settings is not None:
        settings["type"] = kind
    return {"noise": noise_bg, "grass": grass_bg, "asphalt": asphalt_bg, "text": text_bg}[kind](rng, iw, ih, pxmm)


def photo_bg(rng, files, iw, ih, settings=None):
    for _ in range(10):
        path = files[int(rng.integers(len(files)))]
        im = cv2.imread(str(path))
        if im is None:
            continue
        h0, w0 = im.shape[:2]
        cw = int(min(w0, h0 * iw / ih) * rng.uniform(0.3, 1.0))
        ch = int(cw * ih / iw)
        if cw < 16 or ch < 16:
            continue
        x0, y0 = int(rng.integers(0, max(1, w0 - cw + 1))), int(rng.integers(0, max(1, h0 - ch + 1)))
        crop = im[y0:y0 + ch, x0:x0 + cw]
        mirrored = bool(rng.random() < .5)
        if settings is not None:
            settings.update(type="photo", source=path.name,
                            crop_xywh=[x0, y0, cw, ch], mirrored=mirrored)
        return np.ascontiguousarray(cv2.resize(crop[:, ::-1] if mirrored else crop, (iw, ih), interpolation=cv2.INTER_AREA))
    if settings is not None:
        settings["type"] = "noise (photo fallback)"
    return noise_bg(rng, iw, ih)


def draw_confusers(img, rng, n, settings=None):
    """Formas parecidas com partes do alvo (nunca um H) para reduzir falsos positivos."""
    ih, iw = img.shape[:2]
    entries = []
    for _ in range(n):
        kind = ("ring", "ellipse", "rect", "letter", "bars")[int(rng.integers(5))]
        col = tuple(int(v) for v in np.clip(rng.integers(0, 255) + rng.integers(-20, 21, 3), 0, 255))
        c, r = (int(rng.uniform(0, iw)), int(rng.uniform(0, ih))), int(rng.uniform(15, 0.35 * iw))
        th = int(rng.uniform(3, max(4, r * 0.4)))
        entry = dict(kind=kind, center_px=c, radius_px=r, thickness_px=th, color_bgr=col)
        if kind == "ring":
            cv2.circle(img, c, r, col, th, cv2.LINE_AA)
        elif kind == "ellipse":
            ratio = float(rng.uniform(.3, 1))
            angle = float(rng.uniform(0, 180))
            axes = (r, max(4, int(r * ratio)))
            entry.update(axes_px=axes, angle_deg=round(angle, 1))
            cv2.ellipse(img, c, axes, angle, 0, 360, col, th, cv2.LINE_AA)
        elif kind == "rect":
            opposite = (c[0] + int(rng.uniform(-.4, .4) * iw),
                        c[1] + int(rng.uniform(-.4, .4) * ih))
            filled = bool(rng.random() < .5)
            entry.update(opposite_px=opposite, filled=filled)
            cv2.rectangle(img, c, opposite, col, -1 if filled else th)
        elif kind == "letter":
            letter = LETTERS[int(rng.integers(len(LETTERS)))]
            scale = float(rng.uniform(2, 9))
            text_thickness = int(rng.uniform(4, 22))
            entry.update(letter=letter, scale=round(scale, 2), text_thickness_px=text_thickness)
            cv2.putText(img, letter, c, FONT, scale, col, text_thickness, cv2.LINE_AA)
        else:  # duas barras paralelas
            a = rng.uniform(0, math.pi)
            entry["angle_deg"] = round(math.degrees(a), 1)
            dx, dy, nx, ny = math.cos(a) * r * 2, math.sin(a) * r * 2, -math.sin(a) * r * .5, math.cos(a) * r * .5
            for s in (-1, 1):
                cv2.line(img, (int(c[0] - dx + s * nx), int(c[1] - dy + s * ny)),
                         (int(c[0] + dx + s * nx), int(c[1] + dy + s * ny)), col, th, cv2.LINE_AA)
        entries.append(entry)
    if settings is not None:
        settings.extend(entries)


def add_occluder(img, pts, upx, rng, settings=None):
    """Mancha sobre um vértice aleatório; devolve a máscara (para marcar v=1)."""
    ih, iw = img.shape[:2]
    mask = np.zeros((ih, iw), np.uint8)
    cx, cy = pts[int(rng.integers(len(pts)))] + rng.normal(0, 0.4 * upx, 2)
    ax, ay = (int(max(2, v * upx)) for v in rng.uniform(0.4, 1.6, 2))
    is_ellipse = bool(rng.random() < 0.5)
    angle = float(rng.uniform(0, 180)) if is_ellipse else None
    if is_ellipse:
        cv2.ellipse(mask, (int(cx), int(cy)), (ax, ay), angle, 0, 360, 255, -1)
    else:
        cv2.rectangle(mask, (int(cx - ax), int(cy - ay)), (int(cx + ax), int(cy + ay)), 255, -1)
    # Limita cada canal para evitar cores demasiado próximas do branco.
    color = tuple(int(v) for v in rng.integers(0, 200, 3))
    img[mask > 0] = color
    if settings is not None:
        settings.update(shape="ellipse" if is_ellipse else "rectangle",
                        center_px=[round(float(cx), 1), round(float(cy), 1)],
                        size_px=[ax, ay], angle_deg=round(angle, 1) if angle is not None else None,
                        color_bgr=color)
    return mask


def photometric(img, rng, sess, upx=10.0, settings=None, blurred=False):
    """Luz (gradiente, sombra, exposição, gama), desfocagem da lente, ruído do sensor, sem JPEG."""
    ih, iw = img.shape[:2]
    f = img.astype(np.float32) / 255.0
    ang = rng.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[-1:1:ih * 1j, -1:1:iw * 1j]
    gradient = rng.uniform(0, LIGHT_GRADIENT)
    f *= np.clip(1.0 + gradient * (math.cos(ang) * xx + math.sin(ang) * yy), 0.05, None)[..., None].astype(np.float32)
    shadow_settings = None
    if rng.random() < LIGHT_SHADOW_PROB:  # sombra
        sh = np.zeros((ih, iw), np.float32)
        vertices = np.stack([rng.uniform(0, iw, 4), rng.uniform(0, ih, 4)], 1).astype(np.int32)
        cv2.fillConvexPoly(sh, cv2.convexHull(vertices), 1.0)
        opacity, shadow_blur = rng.uniform(0.2, 0.55), float(rng.uniform(3, 25))
        f *= (1.0 - opacity * cv2.GaussianBlur(sh, (0, 0), shadow_blur))[..., None]
        shadow_settings = dict(opacity=round(float(opacity), 3), blur_px=round(shadow_blur, 1),
                               vertices_px=vertices.tolist())
    gain = sess["gain"] * math.exp(rng.normal(0, LIGHT_GAIN_IMG))
    if blurred and BLUR_EXPO_MAX:  # auto-exposição: com blur o branco não satura (senão o clip corta as rampas e o H fica nítido mas deformado)
        gain = min(gain, BLUR_EXPO_MAX / float(np.percentile(f * sess["cast"], 99.5)))
    gamma = rng.uniform(*GAMMA)
    out = (np.clip(f * gain * sess["cast"], 0, 1) ** gamma) * 255
    lens_blur = 0.0
    if rng.random() < 0.35:  # desfocagem (limitada pelo tamanho do alvo)
        lens_blur = float(min(rng.uniform(0.4, 2.0), max(0.5, 0.3 * upx)))
        out = cv2.GaussianBlur(out, (0, 0), lens_blur)
    sg = rng.uniform(0, NOISE_MAX) * 255 / math.sqrt(np.clip(gain, 0.25, 1.5))  # mais ruído com pouca luz
    if settings is not None:
        settings.update(gain=round(float(gain), 3), color_cast=[round(float(v), 3) for v in sess["cast"]],
                        gradient=round(float(gradient), 3), gradient_angle_deg=round(math.degrees(ang), 1),
                        shadow=shadow_settings, gamma=round(float(gamma), 3),
                        lens_blur_sigma_px=round(lens_blur, 2), noise_sigma=round(float(sg), 2))
    return np.clip(out + rng.normal(0, 1, out.shape) * sg, 0, 255).astype(np.uint8)


# ---------------------------------------- labels e overlay ---------------------------------------------
def make_label(pts, inside, occ, ring, iw, ih, v0):
    """Linha YOLO-pose: caixa = envolvente do ANEL (+3%, cortada à imagem); v=0 fora da imagem ou se v0."""
    x0, y0 = ring.min(0)
    x1, y1 = ring.max(0)
    mw, mh = 0.03 * (x1 - x0), 0.03 * (y1 - y0)
    x0, y0, x1, y1 = max(0.0, x0 - mw), max(0.0, y0 - mh), min(float(iw), x1 + mw), min(float(ih), y1 + mh)
    tok = [f"{(x0 + x1) / 2 / iw:.6f}", f"{(y0 + y1) / 2 / ih:.6f}", f"{(x1 - x0) / iw:.6f}", f"{(y1 - y0) / ih:.6f}"]
    vis = []
    for (x, y), ins in zip(pts, inside):
        v = 0 if (v0 or not ins) else (1 if occ is not None and occ[int(round(y)), int(round(x))] > 0 else 2)
        tok += [f"{x / iw:.6f}", f"{y / ih:.6f}", str(v)] if v else ["0.000000", "0.000000", "0"]
        vis.append(v)
    return "0 " + " ".join(tok), vis


def debug_overlay(img, pose, vis, line, K, x_m, info=""):
    """Desenha o que está no LABEL (keypoints v=2 verde, v=1 laranja; v=0 cinzento oco = não anotado, ignorado no treino),
    contorno entre keypoints anotados consecutivos, caixa do label; eixos N/E/D e atitude vêm da pose."""
    o = img.copy()
    ih, iw = o.shape[:2]
    v = [float(s) for s in line.split()]
    kp = np.array(v[5:]).reshape(-1, 3) * (iw, ih, 1)
    for i in range(len(kp)):
        j = (i + 1) % len(kp)
        if kp[i, 2] > 0 and kp[j, 2] > 0:
            cv2.line(o, tuple(int(c) for c in np.round(kp[i, :2])), tuple(int(c) for c in np.round(kp[j, :2])), (0, 255, 255), 1, cv2.LINE_AA)
    for i, (x, y, f) in enumerate(kp):
        c = (int(round(x)), int(round(y)))
        if f > 0:
            cv2.circle(o, c, 3, (0, 255, 0) if f == 2 else (0, 165, 255), -1, cv2.LINE_AA)
            cv2.putText(o, str(i), (c[0] + 4, c[1] - 4), FONT, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(o, c, 4, (160, 160, 160), 1, cv2.LINE_AA)
    bx, by, bw, bh = v[1] * iw, v[2] * ih, v[3] * iw, v[4] * ih
    cv2.rectangle(o, (int(bx - bw / 2), int(by - bh / 2)), (int(bx + bw / 2), int(by + bh / 2)), (255, 0, 0), 1)
    L = 3 * x_m
    ax = project(np.array([[0, 0, 0], [0, -L, 0], [L, 0, 0], [0, 0, L]], dtype=np.float64), pose["rvec"], pose["t"], K)
    o0 = tuple(int(c) for c in np.round(ax[0]))
    for q, col, name in ((ax[1], (0, 0, 255), "N"), (ax[2], (0, 200, 0), "E"), (ax[3], (255, 0, 0), "D")):
        qq = tuple(int(c) for c in np.round(q))
        cv2.arrowedLine(o, o0, qq, col, 2, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(o, name, (qq[0] + 4, qq[1] + 4), FONT, 0.6, col, 2, cv2.LINE_AA)
    a = pose["attitude"]
    txt = f"roll {a['roll']:+.0f} pitch {a['pitch']:+.0f} yaw {a['yaw']:+.0f} | alt {-pose['pos_ned'][2]:.2f} m | {info}"
    (tw, th), _ = cv2.getTextSize(txt, FONT, 0.6, 1)
    cv2.rectangle(o, (6, 6), (14 + tw, 32), (0, 0, 0), -1)
    cv2.putText(o, txt, (10, 24), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return o


# -------------------------------------------- generate -------------------------------------------------
def progress_line(completed, total, elapsed):
    percent = completed / total if total else 1.0
    filled = int(30 * percent)
    remaining = (elapsed * (total - completed) / completed) if completed else None

    def format_duration(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

    eta = format_duration(remaining) if remaining is not None else "calculando"
    return (f"[{'#' * filled}{'-' * (30 - filled)}] {percent:6.1%} "
            f"{completed}/{total} | decorrido {format_duration(elapsed)} "
            f"| restante ~{eta}")


def generate():
    rng = np.random.default_rng(SEED)
    K, iw, ih = load_cam()
    x_m = X_CM / 100.0
    out = Path(OUT_DIR)
    for sub in ("images", "labels", "poses"):
        (out / sub / SPLIT).mkdir(parents=True, exist_ok=True)
    (out / "debug").mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({
        "x_cm": X_CM, "image_size": [iw, ih], "K": K.tolist(), "vertices_units_x": H_UNITS.tolist(),
        "vertices_3d_m": obj_pts(x_m).tolist(), "box": "anel (raio exterior 5x)",
        "sampling": dict(roll_max=ROLL_MAX, pitch_max=PITCH_MAX, yaw_max=YAW_MAX, tilt_cap=TILT_CAP, dmin=DMIN,
                         dmax=DMAX, d_log=D_LOG, blur_max_x=BLUR_MAX_X, blur_v0_above_x=BLUR_V0_ABOVE_X),
        "frames": "heliponto NED (N=cima do H, E=direita, D=chão); corpo FRD; câmara nadir; "
                  "R_ned<-corpo=Rz(yaw)Ry(pitch)Rx(roll); X_cam=R@X_alvo+t, alvo T=(E,-N,D)",
        "seed": SEED}, indent=2))

    bgs = sorted(p for p in Path(BACKGROUNDS).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if BACKGROUNDS else []
    n_neg = int(round(N_IMAGES * NEG_FRAC))
    neg_idx = set(rng.choice(N_IMAGES, n_neg, replace=False).tolist()) if n_neg else set()
    n_pos = N_IMAGES - n_neg
    perms = [rng.permutation(n_pos) for _ in range(4)]  # Latin hypercube: roll, pitch, yaw, distância
    jpos, dbg, cur, sess = 0, 0, -1, None
    st = dict(pos=0, neg=0, fake=0, partial=0, occ=0, blur=0, v0=0, failed=0)
    started_at = time.monotonic()
    last_progress_at = started_at

    def report_progress(completed, force=False):
        nonlocal last_progress_at
        now = time.monotonic()
        if not force and completed < N_IMAGES and now - last_progress_at < 1.0:
            return
        end = completed >= N_IMAGES
        separator = "\n" if end or not sys.stderr.isatty() else "\r"
        print(progress_line(completed, N_IMAGES, now - started_at),
              end=separator, file=sys.stderr, flush=True)
        last_progress_at = now

    report_progress(0, force=True)

    for i in range(N_IMAGES):
        s = i * SESSIONS // N_IMAGES
        if s != cur:
            cur, sess = s, {"gain": math.exp(rng.uniform(math.log(LIGHT_GAIN[0]), math.log(LIGHT_GAIN[1]))),
                            "cast": rng.uniform(1 - COLOR_CAST, 1 + COLOR_CAST, 3).astype(np.float32)}
        name = f"synth{s:02d}__{i:05d}"
        line, pose, vis, upx, blur_x = "", None, None, 10.0, 0.0
        v0 = False
        sampled_pose = None
        settings = {"has_h": i not in neg_idx, "background": {}, "confusers": []}

        if i in neg_idx:
            fk = sample_pose(rng, K, iw, ih, x_m) if rng.random() < HARD_NEG_FRAC else None
            sampled_pose = fk
            d = fk["d"] if fk else math.exp(rng.uniform(math.log(DMIN), math.log(DMAX)))
            settings["hard_negative"] = fk is not None
            bg_kind = "text" if (fk is None and rng.random() < TEXT_NEG_FRAC) else None
            bg = make_bg(rng, bgs, iw, ih, K[0, 0] / d / 1000, bg_kind, settings["background"])
            if fk is None:
                n_confusers = int(rng.integers(0, 4))
                draw_confusers(bg, rng, n_confusers, settings["confusers"])
            else:  # falso alvo: sem label
                fake = str(rng.choice(["noring", "wrongH"]))
                settings["target"] = {}
                bg, upx = render_target(bg, fk, K, iw, ih, x_m, rng, fake=fake,
                                        settings=settings["target"])
                st["fake"] += 1
            settings["blur"] = dict(length_x=0.0, length_px=0.0, angle_deg=None, scope="none")
            settings["occlusion"] = None
            img = photometric(bg, rng, sess, upx, settings.setdefault("photometric", {}))
            st["neg"] += 1
        else:
            strat = tuple(int(perms[k][jpos]) for k in range(4)) + (n_pos,)
            jpos += 1
            partial_requested = rng.random() < PARTIAL_FRAC
            pose = sample_pose(rng, K, iw, ih, x_m, strat, partial_requested) or sample_pose(rng, K, iw, ih, x_m, strat)
            if pose is None:
                st["failed"] += 1
                report_progress(i + 1)
                continue
            sampled_pose = pose
            bg = make_bg(rng, bgs, iw, ih, K[0, 0] / pose["d"] / 1000,
                         settings=settings["background"])
            if rng.random() < 0.5:
                n_confusers = int(rng.integers(1, 4))
                draw_confusers(bg, rng, n_confusers, settings["confusers"])
            upx = K[0, 0] * x_m / pose["d"]
            kern = None
            blur_angle = None
            if rng.random() < BLUR_PROB:  # motion blur linear: comprimento em unidades x, direção aleatória
                blur_x = min(BLUR_MAX_X * rng.random() ** 2, BLUR_MAX_PX / upx)
                blur_angle = float(rng.uniform(0, 180))
                kern = motion_kernel(blur_x * upx, blur_angle)
            whole = kern is not None and rng.random() >= BLUR_TARGET_ONLY  # imagem toda (câmara) ou só o alvo
            settings["blur"] = dict(length_x=round(float(blur_x), 3),
                                    length_px=round(float(blur_x * upx), 2),
                                    angle_deg=round(blur_angle, 1) if blur_angle is not None else None,
                                    scope=("whole image" if whole else "target") if kern is not None else "none")
            settings["target"] = {}
            img, upx = render_target(bg, pose, K, iw, ih, x_m, rng,
                                     kernel=None if whole else kern, settings=settings["target"])
            if whole:
                img = cv2.filter2D(img, -1, kern, borderType=cv2.BORDER_REFLECT)
            settings["occlusion"] = {}
            occ = add_occluder(img, pose["pts"], upx, rng, settings["occlusion"]) if rng.random() < OCC_PROB else None
            if occ is None:
                settings["occlusion"] = None
            v0 = BLUR_V0_ABOVE_X is not None and blur_x > BLUR_V0_ABOVE_X
            line, vis = make_label(pose["pts"], pose["inside"], occ, pose["ring"], iw, ih, v0)
            img = photometric(img, rng, sess, upx, settings.setdefault("photometric", {}), kern is not None)
            st["pos"] += 1
            st["partial"] += int(pose["inside"].sum() < NUM_KPTS)
            st["occ"] += int(occ is not None and 1 in vis)
            st["blur"] += int(blur_x * upx >= 1.0)
            st["v0"] += int(v0)

        if sampled_pose is not None:
            settings["pose"] = dict(
                attitude_deg=sampled_pose["attitude"],
                distance_m=round(float(sampled_pose["d"]), 3),
                tilt_deg=round(float(sampled_pose["tilt"]), 1),
                center_px=np.round(sampled_pose["pts"].mean(axis=0), 1).tolist(),
                pad_shape=sampled_pose["shape"],
                pad_size_units=np.round(sampled_pose["pad_size_units"], 2).tolist(),
                pad_geometry=sampled_pose["shape_details"],
                partial_requested=sampled_pose["partial_requested"],
                keypoints_inside=int(sampled_pose["inside"].sum()),
                keypoints_visible=int(np.count_nonzero(np.array(vis) > 0)) if vis is not None else None,
                keypoints_zeroed_by_blur=bool(v0),
                relaxed=sampled_pose["relaxed"])
        quality = int(rng.integers(40, 96))
        settings["jpeg_quality"] = quality
        cv2.imwrite(str(out / "images" / SPLIT / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        (out / "labels" / SPLIT / f"{name}.txt").write_text(line + ("\n" if line else ""))
        # (out / "settings" / SPLIT / f"{name}.json").write_text(json.dumps(settings))
        if pose is not None:
            R, t = pose["R"], pose["t"]
            (out / "poses" / SPLIT / f"{name}.json").write_text(json.dumps({
                "image": f"{name}.jpg", "session": f"synth{s:02d}", "attitude_deg": pose["attitude"],
                "position_ned_m": pose["pos_ned"].tolist(), "altitude_m": float(-pose["pos_ned"][2]),
                "R": R.tolist(), "rvec": pose["rvec"].ravel().tolist(), "t_m": t.tolist(),
                "camera_position_in_target_m": (-R.T @ t).tolist(), "distance_m": float(np.linalg.norm(t)),
                "tilt_deg": pose["tilt"], "n_visible_kpts": int((np.array(vis) > 0).sum()), "pad_shape": pose["shape"],
                "blur_x": round(blur_x, 3), "blur_px": round(blur_x * upx, 2), "relaxed": pose["relaxed"],
                "gt_source": "sintetico-opencv"}))
            if dbg < DEBUG_N:
                cv2.imwrite(str(out / "debug" / f"{name}.jpg"),
                            debug_overlay(img, pose, vis, line, K, x_m, f"{pose['shape']} blur {blur_x:.2f}x"))
                dbg += 1
        report_progress(i + 1)
    print(f"Gerado em {out}: {st['pos']} com H ({st['partial']} parciais, {st['occ']} com oclusão, {st['blur']} com blur "
          f"≥1 px, {st['v0']} com keypoints v=0 por blur), {st['neg']} sem H ({st['fake']} com falso alvo), {st['failed']} falhadas.")


# ---------------------------------------- verify (debug) -----------------------------------------------
def hist_report(title, vals, lo, hi, nb, log=False, unit=""):
    vals = np.asarray(vals, dtype=float)
    edges = np.geomspace(lo, hi, nb + 1) if log else np.linspace(lo, hi, nb + 1)
    cnt, _ = np.histogram(vals, bins=edges)
    print(f"  {title}")
    for b in range(nb):
        print(f"    {edges[b]:8.2f} .. {edges[b + 1]:8.2f} {unit:3s}| {cnt[b]:5d} {'#' * int(36 * cnt[b] / max(cnt.max(), 1))}")
    return cnt.min() >= 0.5 * len(vals) / nb


def verify():
    out = Path(OUT_DIR)
    meta = json.loads((out / "meta.json").read_text())
    K, (iw, ih) = np.array(meta["K"]), meta["image_size"]
    P, x_m, smp = np.array(meta["vertices_3d_m"]), meta["x_cm"] / 100.0, meta["sampling"]
    RP = ring_pts(x_m)
    poses = sorted((out / "poses" / SPLIT).glob("*.json"))
    if not poses:
        print(f"Sem poses em {out / 'poses' / SPLIT}")
        return
    rows = []
    for pj in poses:
        js = json.loads(pj.read_text())
        name = Path(js["image"]).stem
        lines = [l for l in (out / "labels" / SPLIT / f"{name}.txt").read_text().splitlines() if l.strip()]
        if not lines:
            continue
        v = np.array(lines[0].split(), dtype=float)
        kp = v[5:].reshape(-1, 3)
        px, vis = kp[:, :2] * [iw, ih], kp[:, 2] > 0
        R, t, rvec = np.array(js["R"]), np.array(js["t_m"]), np.array(js["rvec"]).reshape(3, 1)
        r = dict(js=js, name=name, e_rep=float("nan"), e_rot=float("nan"), e_t=float("nan"))
        att = R_to_att(R)  # 1. JSON coerente
        r["e_json"] = max(max(abs(wrap180(att[k] - js["attitude_deg"][k])) for k in att),
                          float(np.abs(pose_to_ned(R, t) - np.array(js["position_ned_m"])).max()) * 1000)
        ring = project(RP, rvec, t, K)  # 2. caixa do label = envolvente do anel (+3%)
        x0, y0, x1, y1 = (ring.min(0)[0], ring.min(0)[1], ring.max(0)[0], ring.max(0)[1])
        mw, mh = .03 * (x1 - x0), .03 * (y1 - y0)
        exp = np.array([max(0, x0 - mw), max(0, y0 - mh), min(iw, x1 + mw), min(ih, y1 + mh)])
        got = np.array([(v[1] - v[3] / 2) * iw, (v[2] - v[4] / 2) * ih, (v[1] + v[3] / 2) * iw, (v[2] + v[4] / 2) * ih])
        r["e_box"] = float(np.abs(exp - got).max())
        if vis.sum() >= 1:  # 3. reprojeção dos keypoints anotados
            r["e_rep"] = float(np.linalg.norm(project(P, rvec, t, K)[vis] - px[vis], axis=1).max())
        if vis.sum() >= 4:  # 4. ida e volta no PnP
            sol = solve_pose(px[vis], K, None, P[vis])
            if sol is not None:
                r["e_rot"], r["e_t"] = rot_err_deg(cv2.Rodrigues(sol[0])[0], R), float(np.linalg.norm(sol[1] - t) * 1000)
        r["x_px"] = K[0, 0] * x_m / t[2]
        rows.append(r)
    n = len(rows)
    a = lambda k: np.array([r[k] for r in rows])
    ok_rot = a("e_rot")[~np.isnan(a("e_rot")) & (a("e_rot") <= TOL_ROT_DEG)]
    n_pnp = int((~np.isnan(a("e_rot"))).sum())
    ambig = int(np.nansum(a("e_rot") > AMBIG_DEG))
    bad_rep = int(np.nansum(a("e_rep") > TOL_REPROJ_PX))
    bad_json = int((a("e_json") > 1e-3).sum())
    bad_box = int((a("e_box") > 0.05).sum())
    bad_num = int(np.nansum((a("e_rot") > TOL_ROT_DEG) & (a("e_rot") <= AMBIG_DEG)))
    print(f"VERIFICAÇÃO ({n} imagens com H em {SPLIT}; {n - n_pnp} sem keypoints anotados por blur/fora)")
    print(f"  1. JSON coerente (atitude/posição recalculadas de R,t): {n - bad_json}/{n}")
    print(f"  2. Caixa do label = envolvente do anel projetado: máx {np.nanmax(a('e_box')):.2f} px; falhas {bad_box}")
    print(f"  3. Reprojeção da pose vs keypoints anotados: máx {np.nanmax(a('e_rep')):.4f} px (limite {TOL_REPROJ_PX}); falhas {bad_rep}")
    print(f"  4. PnP ida e volta ({n_pnp} imagens): {len(ok_rot)} exatas; solução errada do plano (>{AMBIG_DEG:.0f}°): {ambig}; "
          f"falhas numéricas: {bad_num}")
    bl = np.array([r["js"]["blur_px"] for r in rows])
    print(f"  Motion blur: {int((bl >= 1).sum())} imagens com ≥1 px (máx {bl.max():.1f} px); "
          f"formas do pad: " + ", ".join(f"{s}={sum(r['js']['pad_shape'] == s for r in rows)}" for s in SHAPES))
    print("\nCOBERTURA")
    good = True
    for k, m in (("roll", smp["roll_max"]), ("pitch", smp["pitch_max"]), ("yaw", smp["yaw_max"])):
        good &= hist_report(f"{k} (°)", [r["js"]["attitude_deg"][k] for r in rows], -m, m, 8, unit="deg")
    good &= hist_report("distância ao centro do H (m)", [r["js"]["distance_m"] for r in rows], smp["dmin"] * .99,
                        smp["dmax"] * 1.01, 8, log=smp["d_log"], unit="m")
    for sz in (640, 1280):
        print(f"  H com <35 px de largura na entrada da rede (imgsz={sz}): "
              f"{100 * np.mean(4 * a('x_px') * sz / iw < 35):.0f}% das imagens; anel com <35 px: "
              f"{100 * np.mean(10 * a('x_px') * sz / iw < 35):.0f}%")
    ok = bad_json == 0 and bad_box == 0 and bad_rep == 0 and bad_num == 0
    print(f"\n{'OK' if ok else 'COM PROBLEMAS'}" + ("" if good else " (cobertura fraca numa faixa: ver acima)"))


if __name__ == "__main__":
    if MODE == "generate":
        generate()
    verify()