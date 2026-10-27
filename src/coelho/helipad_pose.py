import numpy as np
import cv2 as cv

# ---------- Modelo 3D do H (medido no alvo impresso, em metros) ----------
W, HT, T, C = 0.066, 0.0825, 0.0165, 0.0165

def h_model(W=W, HT=HT, T=T, C=C):
    """12 vértices, origem no centro, X direita, Y baixo, Z=0.
    Ordem K1..K12: canto sup. esq., sentido horário (contrato com a Bea e o Abanico)."""
    w, h, c = W / 2, HT / 2, C / 2
    pts = [(-w, -h), (-w + T, -h), (-w + T, -c), (w - T, -c),
           (w - T, -h), (w, -h), (w, h), (w - T, h),
           (w - T, c), (-w + T, c), (-w + T, h), (-w, h)]
    return np.array([(x, y, 0.0) for x, y in pts], np.float32)

H_MODEL = h_model()

# ---------- Auxiliares ----------
def rot_error_deg(Ra, Rb):
    """Ângulo da rotação relativa entre duas matrizes de rotação."""
    cosang = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

def euler_zyx_deg(R):
    """Roll, pitch, yaw (graus), convenção ZYX: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    Só para apresentação. Para ERROS usa rot_error_deg."""
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:                                  # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])

def _reproj_rmse(obj, img, rvec, tvec, K, dist):
    proj, _ = cv.projectPoints(obj, rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - img
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))

# ---------- Função principal ----------
def estimate_pose(pts2d, conf, K, dist, prev=None, obj=None,
                  conf_thr=0.5, min_pts=4):
    """
    pts2d : (12,2) px da imagem original (NaN nos pontos em falta)
    conf  : (12,)  confiança 0..1
    prev  : resultado anterior (dict) para consistência temporal, ou None
    Devolve um dict; 'success' diz se há pose válida.
    """
    obj = H_MODEL if obj is None else obj
    pts2d = np.asarray(pts2d, np.float32).reshape(-1, 2)
    conf = np.asarray(conf, np.float32).reshape(-1)

    valid = (np.isfinite(pts2d).all(axis=1) & np.isfinite(conf)
             & (conf >= conf_thr))
    out = {"success": False, "n_valid": int(valid.sum())}
    if valid.sum() < min_pts:
        return out

    o = np.ascontiguousarray(obj[valid])
    p = np.ascontiguousarray(pts2d[valid])

    # 1) IPPE devolve as 2 soluções do alvo planar
    n, rvecs, tvecs, _ = cv.solvePnPGeneric(o, p, K, dist,
                                            flags=cv.SOLVEPNP_IPPE)
    if n == 0:
        return out

    # 2) refinar cada candidata e medir erro de reprojeção
    cands = []
    for r, t in zip(rvecs, tvecs):
        r, t = cv.solvePnPRefineLM(o, p, K, dist, r.copy(), t.copy())
        if t.ravel()[2] > 0:                       # alvo à frente da câmara
            cands.append((r, t, _reproj_rmse(o, p, r, t, K, dist)))
    if not cands:
        return out

    # 3) escolher: menor reprojeção, ou mais próxima do frame anterior
    best = min(range(len(cands)), key=lambda i: cands[i][2])
    if prev is not None and prev.get("success") and len(cands) > 1:
        Rp = prev["R"]
        near = min(range(len(cands)),
                   key=lambda i: rot_error_deg(Rp, cv.Rodrigues(cands[i][0])[0]))
        # só aceita a vizinha se não for muito pior em reprojeção
        if cands[near][2] <= 2 * cands[best][2] + 1.0:
            best = near

    r, t, rp = cands[best]
    R, _ = cv.Rodrigues(r)
    alt = [c[2] for i, c in enumerate(cands) if i != best]
    out.update(success=True, rvec=r.ravel(), tvec=t.ravel(), R=R,
               euler_deg=euler_zyx_deg(R),            # roll, pitch, yaw do alvo na câmara
               cam_pos=(-R.T @ t).ravel(),            # câmara no referencial do alvo
               reproj_px=rp,
               alt_reproj_px=alt[0] if alt else None)  # perto de reproj_px = ambíguo
    return out

def get_candidates(pts2d, conf, K, dist, obj=None, conf_thr=0.5, min_pts=4):
    """Devolve as soluções IPPE refinadas (lista de dicts), sem escolher."""
    obj = H_MODEL if obj is None else obj
    pts2d = np.asarray(pts2d, np.float32).reshape(-1, 2)
    conf = np.asarray(conf, np.float32).reshape(-1)
    valid = (np.isfinite(pts2d).all(axis=1) & np.isfinite(conf)
             & (conf >= conf_thr))
    if valid.sum() < min_pts:
        return []
    o = np.ascontiguousarray(obj[valid])
    p = np.ascontiguousarray(pts2d[valid])
    n, rvecs, tvecs, _ = cv.solvePnPGeneric(o, p, K, dist,
                                            flags=cv.SOLVEPNP_IPPE)
    cands = []
    for r, t in zip(rvecs, tvecs):
        r, t = cv.solvePnPRefineLM(o, p, K, dist, r.copy(), t.copy())
        if t.ravel()[2] > 0:
            cands.append({"rvec": r.ravel(), "tvec": t.ravel(),
                          "R": cv.Rodrigues(r)[0],
                          "reproj_px": _reproj_rmse(o, p, r, t, K, dist)})
    return cands

def track_viterbi(cands_seq, lam=0.02):
    """cands_seq: lista (um elemento por frame) de listas de candidatos.
    Devolve, por frame, o índice da candidata escolhida (None se não houver).
    lam: custo em px por grau de salto de rotação entre frames."""
    idx = [i for i, c in enumerate(cands_seq) if c]
    choice = [None] * len(cands_seq)
    if not idx:
        return choice
    cost = [c["reproj_px"] for c in cands_seq[idx[0]]]
    back = []
    for a, b in zip(idx[:-1], idx[1:]):
        new, bp = [], []
        for cb in cands_seq[b]:
            tr = [cost[j] + lam * rot_error_deg(ca["R"], cb["R"])
                  for j, ca in enumerate(cands_seq[a])]
            j = int(np.argmin(tr))
            bp.append(j)
            new.append(tr[j] + cb["reproj_px"])
        cost = new
        back.append(bp)
    k = int(np.argmin(cost))
    for f in range(len(idx) - 1, -1, -1):
        choice[idx[f]] = k
        if f > 0:
            k = back[f - 1][k]
    return choice