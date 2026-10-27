import numpy as np, cv2 as cv
from pathlib import Path
from helipad_pose import (H_MODEL, estimate_pose, get_candidates,
                          track_viterbi, rot_error_deg)

cam = np.load(Path(__file__).resolve().parent / "camera.npz")
K, dist = cam["K"], cam["dist"]

def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

rng = np.random.default_rng(1)
N_TRAJ, N_FR, SIGMA = 20, 100, 1.0
LAMS = [0.005, 0.02, 0.05, 0.1]

total = 0
f_base = 0
f_greedy = 0
f_vit = {lam: 0 for lam in LAMS}
greedy_per_traj = []

for _ in range(N_TRAJ):
    prev, g_fail = None, 0
    cands_seq, R_gts = [], []
    for k in range(N_FR):
        s = k / (N_FR - 1)
        R_gt = (Rz(np.radians(30 * s)) @ Rx(np.radians(25 * np.sin(2*np.pi*s)))
                @ Ry(np.radians(20 * np.cos(2*np.pi*s))))
        t_gt = np.array([0.05*np.sin(2*np.pi*s), 0.02, 1.5 - 0.9*s])
        rvec_gt, _ = cv.Rodrigues(R_gt)

        proj, _ = cv.projectPoints(H_MODEL, rvec_gt, t_gt, K, dist)
        pts = proj.reshape(-1, 2) + rng.normal(0, SIGMA, (12, 2))
        conf = np.ones(12, np.float32)
        for i in rng.choice(12, rng.integers(0, 3), replace=False):
            pts[i], conf[i] = np.nan, 0.0

        # método antigo (ganancioso) para comparar
        rb = estimate_pose(pts, conf, K, dist, prev=prev)
        prev = rb if rb["success"] else None
        bad = (not rb["success"]) or rot_error_deg(R_gt, rb["R"]) > 10
        g_fail += bad
        f_greedy += bad

        c = get_candidates(pts, conf, K, dist)
        cands_seq.append(c)
        R_gts.append(R_gt)
        total += 1

        # baseline: menor reprojeção, frame a frame
        if not c:
            f_base += 1
        else:
            b = min(c, key=lambda x: x["reproj_px"])
            f_base += rot_error_deg(R_gt, b["R"]) > 10

    greedy_per_traj.append(g_fail)
    for lam in LAMS:
        ch = track_viterbi(cands_seq, lam)
        for c, k, R_gt in zip(cands_seq, ch, R_gts):
            f_vit[lam] += (k is None) or rot_error_deg(R_gt, c[k]["R"]) > 10

print(f"baseline (reproj, frame a frame): {f_base}/{total} ({100*f_base/total:.1f}%)")
print(f"ganancioso (antigo):              {f_greedy}/{total} ({100*f_greedy/total:.1f}%)")
for lam in LAMS:
    print(f"Viterbi lam={lam:<5}:               {f_vit[lam]}/{total} ({100*f_vit[lam]/total:.1f}%)")
print("falhas por trajetória (ganancioso):", greedy_per_traj)