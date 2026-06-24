"""
Two specific checks requested:
  1) Is the dominant "local rhythm motif" of the couplets spread across multiple
     recordings, or does almost all of it come from ONE session?
  2) The coupling of session 2026-06-05 14:59 looks bimodal by eye but the
     single-focus check did NOT flag it. Why? Is it really bimodal?

Reuses the same functions from host/dashboard.py (no reimplementation of the pipeline).
Run from the repo root:  python3 host/analyze_motif_bimodality.py
"""
import glob
import os
from collections import Counter, defaultdict

import numpy as np

import dashboard as D   # is host/ cwd-relative? no: we import it as a module


def build_sessions():
    """Replicates the filter/load of main() but keeps only what is needed:
    label, coupling_ms, ecg_path, couplets."""
    ecg_files = sorted(glob.glob("logs/ecg_*.csv"))
    ecg_files = [f for f in ecg_files
                 if os.path.getsize(f) > D.MIN_FILE_SIZE_MB * 1_000_000]
    ecg_files = [f for f in ecg_files
                 if D.label_from_path(f)[1] not in D.SKIP_SESSIONS]
    sessions = []
    for ecg_path in ecg_files:
        label, base = D.label_from_path(ecg_path)
        data = D.load_session(ecg_path)
        if data is None:
            continue
        t_ecg, vf_arr, peaks, excl = data
        if not peaks:
            continue
        traces_raw = D.collect_traces(t_ecg, vf_arr, peaks, kind="pvc")
        if traces_raw.shape[0] < D.MIN_PVC_COUNT:
            continue
        coupling_ms = []
        for i, p in enumerate(peaks):
            if p["cls"] != "pvc" or i == 0:
                continue
            rr = (p["t"] - peaks[i-1]["t"]) * 1000
            if 200 < rr < 800:
                coupling_ms.append(rr)
        sessions.append({
            "label": label, "ecg_path": ecg_path,
            "coupling_ms": np.array(coupling_ms),
            "couplets": D.find_all_couplets(t_ecg, vf_arr, peaks, excl),
        })
        print(f"  loaded {label}: {len(coupling_ms)} couplings, "
              f"{len(sessions[-1]['couplets'])} couplets")
    return sessions


def q1_motif_by_session(sessions):
    print("\n" + "=" * 70)
    print("Q1  Local rhythm motif: spread across recordings or from a single session?")
    print("=" * 70)
    # motif -> Counter per session
    motif_sess = defaultdict(Counter)
    motif_tot = Counter()
    for s in sessions:
        for c in s["couplets"]:
            m = c.get("ctx_disp", "")
            motif_sess[m][s["label"]] += 1
            motif_tot[m] += 1
    n_coup = sum(motif_tot.values())
    print(f"\nTotal couplets: {n_coup}  ·  distinct motifs: {len(motif_tot)}")

    for motif, tot in motif_tot.most_common(6):
        by = motif_sess[motif]
        n_sess = len(by)
        top_lab, top_n = by.most_common(1)[0]
        share = 100 * top_n / tot
        print(f"\n  motif  {motif}")
        print(f"    occurrences: {tot}  ·  present in {n_sess}/{len(sessions)} sessions"
              f"  ·  dominant session: {top_lab} ({top_n}, {share:.0f}%)")
        for lab, n in by.most_common():
            bar = "#" * n
            print(f"      {lab:>16}  {n:2d}  {bar}")

    # focus on the dominant motif
    dom, dom_n = motif_tot.most_common(1)[0]
    by = motif_sess[dom]
    n_sess = len(by)
    conc = 100 * by.most_common(1)[0][1] / dom_n
    print("\n  --> Q1 VERDICT:")
    if n_sess == 1:
        print(f"      The dominant motif ({dom_n} couplets) comes from a SINGLE session "
              f"({by.most_common(1)[0][0]}).")
    elif conc >= 70:
        print(f"      The dominant motif is CONCENTRATED: {conc:.0f}% from "
              f"{by.most_common(1)[0][0]} (still present in {n_sess} sessions).")
    else:
        print(f"      The dominant motif is DISTRIBUTED across {n_sess} sessions "
              f"(max {conc:.0f}% from a single one). It is a cross-recording pattern.")


def kde_peaks(x, lo=250, hi=750, n=400, prom_frac=0.05):
    """Counts the peaks of the KDE (visual hint of bimodality)."""
    from scipy.stats import gaussian_kde
    x = np.asarray(x, float)
    x = x[(x > 200) & (x < 800)]
    if len(x) < 30:
        return 0, None, None
    kde = gaussian_kde(x)
    grid = np.linspace(lo, hi, n)
    d = kde(grid)
    # internal peaks with minimum prominence
    peaks = []
    thr = d.max() * prom_frac
    for i in range(1, n - 1):
        if d[i] > d[i-1] and d[i] >= d[i+1] and d[i] > thr:
            peaks.append(grid[i])
    return len(peaks), grid, d


def q2_bimodality(sessions):
    print("\n" + "=" * 70)
    print("Q2  06-05 14:59 looks bimodal but was not flagged. Verify.")
    print("=" * 70)
    targets = ["2026-06-05 14:59", "2026-06-05 18:29", "2026-06-06 15:08"]
    by_label = {s["label"]: s for s in sessions}
    for lab in targets:
        s = by_label.get(lab)
        if s is None:
            print(f"\n  {lab}: NOT found"); continue
        c = s["coupling_ms"]
        c = c[(c > 200) & (c < 800)]
        mod = D.coupling_modality(c)
        npk, _, _ = kde_peaks(c)
        flagged = "18:29" in lab or "15:08" in lab
        print(f"\n  {lab}   (n={len(c)})   {'[FLAGGED bimodal]' if flagged else '[not flagged]'}")
        print(f"    median={np.median(c):.0f} ms   KDE peaks={npk}")
        if mod["ok"]:
            mu = mod["mu"]
            print(f"    GMM dBIC(1vs2)={mod['dbic']:+.1f}  (>0 favors 2 comp)")
            print(f"    GMM 2 means: {mu[0]:.0f} / {mu[1]:.0f} ms  "
                  f"(distance {mu[1]-mu[0]:.0f} ms)")
            print(f"    real internal valley (current criterion): {mod['bimodal']}"
                  + (f"  @ {mod['valley']:.0f} ms" if mod["valley"] else ""))
        else:
            print("    GMM: insufficient data")
        # if the two GMM modes exist, try the morphological split at the midpoint
        if mod["ok"] and mod["mu"]:
            valley = mod["valley"] if mod["valley"] else float(np.mean(mod["mu"]))
            morph = D.coupling_focus_morph(s["ecg_path"], valley)
            if morph:
                print(f"    split @ {valley:.0f} ms -> n_lo={morph['n_lo']} n_hi={morph['n_hi']}"
                      f"  QRS template r={morph['corr']:.3f}  "
                      f"(w {morph['w_lo']:.0f}/{morph['w_hi']:.0f} ms)")
                print(f"      -> {'SAME focus (high r)' if morph['corr']>0.97 else 'DIFFERENT morphology, worth a look'}")

    print("\n  --> Note: the current criterion flags 'bimodal' only if the mixture-2")
    print("      has a real INTERNAL VALLEY AND dBIC>0. If 14:59 has a 2-peak KDE but")
    print("      the GMM finds no internal valley, it is a 'shoulder'/asymmetry, not two")
    print("      separate modes. The numbers above say which of the two cases we are in.")


def main():
    print("Loading the sessions (same pipeline as dashboard.py)...")
    sessions = build_sessions()
    if not sessions:
        print("No sessions."); return
    q1_motif_by_session(sessions)
    q2_bimodality(sessions)


if __name__ == "__main__":
    main()
