import numpy as np
import cv2 as cv

# ---------- Modelo 3D do H (medido no alvo impresso, em metros) ----------
W, HT, T, C = 0.066, 0.0825, 0.0165, 0.0165

def h_model(W=W, HT=HT, T=T, C=C):
    """Cria os 12 pontos 3D do H num plano (Z=0).
    Ordem dos cantos: do canto sup. esq. no sentido horário."""
    w, h, c = W / 2, HT / 2, C / 2
    pts = [(-w, -h), (-w + T, -h), (-w + T, -c), (w - T, -c),
           (w - T, -h), (w, -h), (w, h), (w - T, h),
           (w - T, c), (-w + T, c), (-w + T, h), (-w, h)]
    return np.array([(x, y, 0.0) for x, y in pts], np.float32)

H_MODEL = h_model()

# ---------- Funções Auxiliares ----------
def rot_error_deg(Ra, Rb):
    """Calcula o ângulo de erro de rotação entre duas matrizes."""
    cosang = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

def euler_zyx_deg(R):
    """Converte matriz de rotação em ângulos de Euler (Roll, Pitch, Yaw) em graus."""
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:                                  # Evita bloqueio mecânico (gimbal lock)
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])

def _reproj_rmse(obj, img, rvec, tvec, K, dist):
    """Calcula o erro médio de reprojeção em pixéis."""
    proj, _ = cv.projectPoints(obj, rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - img
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))

# ---------- Função principal de estimativa de pose ----------
def estimate_pose(pts2d, conf, K, dist, prev=None, obj=None,
                  conf_thr=0.5, min_pts=4):
    """Estima a pose 3D do alvo a partir dos pontos 2D detetados e respetivas confianças."""
    obj = H_MODEL if obj is None else obj
    pts2d = np.asarray(pts2d, np.float32).reshape(-1, 2)
    conf = np.asarray(conf, np.float32).reshape(-1)

    # Filtrar apenas pontos válidos que cumprem o limiar de confiança
    valid = (np.isfinite(pts2d).all(axis=1) & np.isfinite(conf)
             & (conf >= conf_thr))
    out = {"success": False, "n_valid": int(valid.sum())}
    if valid.sum() < min_pts:
        return out

    o = np.ascontiguousarray(obj[valid])
    p = np.ascontiguousarray(pts2d[valid])

    # 1) O IPPE devolve duas soluções possíveis para alvos planos
    n, rvecs, tvecs, _ = cv.solvePnPGeneric(o, p, K, dist,
                                            flags=cv.SOLVEPNP_IPPE)
    if n == 0:
        return out

    # 2) Refinar cada solução candidata e calcular o erro de reprojeção
    cands = []
    for r, t in zip(rvecs, tvecs):
        r, t = cv.solvePnPRefineLM(o, p, K, dist, r.copy(), t.copy())
        if t.ravel()[2] > 0:                       # Garantir que o alvo está à frente da câmara
            cands.append((r, t, _reproj_rmse(o, p, r, t, K, dist)))
    if not cands:
        return out

    # 3) Escolher a melhor: menor erro ou mais próxima do frame anterior (suavização temporal)
    best = min(range(len(cands)), key=lambda i: cands[i][2])
    if prev is not None and prev.get("success") and len(cands) > 1:
        Rp = prev["R"]
        near = min(range(len(cands)),
                   key=lambda i: rot_error_deg(Rp, cv.Rodrigues(cands[i][0])[0]))
        # Aceita a solução vizinha se o erro de reprojeção não for muito pior
        if cands[near][2] <= 2 * cands[best][2] + 1.0:
            best = near

    r, t, rp = cands[best]
    R, _ = cv.Rodrigues(r)
    alt = [c[2] for i, c in enumerate(cands) if i != best]
    out.update(success=True, rvec=r.ravel(), tvec=t.ravel(), R=R,
               euler_deg=euler_zyx_deg(R),            # Orientação em Euler (Roll, Pitch, Yaw)
               cam_pos=(-R.T @ t).ravel(),            # Posição da câmara no referencial do alvo
               reproj_px=rp,
               alt_reproj_px=alt[0] if alt else None)  # Erro da solução alternativa (indica ambiguidade se perto)
    return out

def get_candidates(pts2d, conf, K, dist, obj=None, conf_thr=0.5, min_pts=4):
    """Devolve todas as soluções candidatas refinadas do IPPE, sem escolher a melhor."""
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
    """Aplica o algoritmo de Viterbi para escolher a melhor sequência de poses 
    ao longo do tempo, minimizando o erro de reprojeção e saltos bruscos de rotação."""
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