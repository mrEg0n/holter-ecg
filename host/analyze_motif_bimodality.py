"""
Due verifiche puntuali chieste:
  1) Il "local rhythm motif" dominante dei couplet e' sparso su piu' registrazioni
     o viene quasi tutto da UNA sessione?
  2) Il coupling della sessione 2026-06-05 14:59 sembra bimodale a occhio ma il
     check single-focus NON l'ha segnalata. Perche'? E' davvero bimodale?

Riusa le stesse funzioni di host/dashboard.py (nessuna ricodifica della pipeline).
Eseguire dalla root del repo:  python3 host/analyze_motif_bimodality.py
"""
import glob
import os
from collections import Counter, defaultdict

import numpy as np

import dashboard as D   # host/ e' la cwd-relative? no: lo importiamo come modulo


def build_sessions():
    """Replica il filtro/carico di main() ma tiene solo cio' che serve:
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
    print("Q1  Local rhythm motif: sparso tra registrazioni o da una sessione?")
    print("=" * 70)
    # motivo -> Counter per sessione
    motif_sess = defaultdict(Counter)
    motif_tot = Counter()
    for s in sessions:
        for c in s["couplets"]:
            m = c.get("ctx_disp", "")
            motif_sess[m][s["label"]] += 1
            motif_tot[m] += 1
    n_coup = sum(motif_tot.values())
    print(f"\nTotale couplet: {n_coup}  ·  motivi distinti: {len(motif_tot)}")

    for motif, tot in motif_tot.most_common(6):
        by = motif_sess[motif]
        n_sess = len(by)
        top_lab, top_n = by.most_common(1)[0]
        share = 100 * top_n / tot
        print(f"\n  motif  {motif}")
        print(f"    occorrenze: {tot}  ·  presente in {n_sess}/{len(sessions)} sessioni"
              f"  ·  sessione dominante: {top_lab} ({top_n}, {share:.0f}%)")
        for lab, n in by.most_common():
            bar = "#" * n
            print(f"      {lab:>16}  {n:2d}  {bar}")

    # focus sul motivo dominante
    dom, dom_n = motif_tot.most_common(1)[0]
    by = motif_sess[dom]
    n_sess = len(by)
    conc = 100 * by.most_common(1)[0][1] / dom_n
    print("\n  --> VERDETTO Q1:")
    if n_sess == 1:
        print(f"      Il motivo dominante ({dom_n} couplet) viene da UNA SOLA sessione "
              f"({by.most_common(1)[0][0]}).")
    elif conc >= 70:
        print(f"      Il motivo dominante e' CONCENTRATO: {conc:.0f}% da "
              f"{by.most_common(1)[0][0]} (presente comunque in {n_sess} sessioni).")
    else:
        print(f"      Il motivo dominante e' DISTRIBUITO su {n_sess} sessioni "
              f"(max {conc:.0f}% da una sola). E' un pattern trans-registrazione.")


def kde_peaks(x, lo=250, hi=750, n=400, prom_frac=0.05):
    """Conta i picchi della KDE (indizio visivo di bimodalita')."""
    from scipy.stats import gaussian_kde
    x = np.asarray(x, float)
    x = x[(x > 200) & (x < 800)]
    if len(x) < 30:
        return 0, None, None
    kde = gaussian_kde(x)
    grid = np.linspace(lo, hi, n)
    d = kde(grid)
    # picchi interni con prominenza minima
    peaks = []
    thr = d.max() * prom_frac
    for i in range(1, n - 1):
        if d[i] > d[i-1] and d[i] >= d[i+1] and d[i] > thr:
            peaks.append(grid[i])
    return len(peaks), grid, d


def q2_bimodality(sessions):
    print("\n" + "=" * 70)
    print("Q2  06-05 14:59 sembra bimodale ma non e' stata segnalata. Verifica.")
    print("=" * 70)
    targets = ["2026-06-05 14:59", "2026-06-05 18:29", "2026-06-06 15:08"]
    by_label = {s["label"]: s for s in sessions}
    for lab in targets:
        s = by_label.get(lab)
        if s is None:
            print(f"\n  {lab}: NON trovata"); continue
        c = s["coupling_ms"]
        c = c[(c > 200) & (c < 800)]
        mod = D.coupling_modality(c)
        npk, _, _ = kde_peaks(c)
        flagged = "18:29" in lab or "15:08" in lab
        print(f"\n  {lab}   (n={len(c)})   {'[FLAGGED bimodale]' if flagged else '[non flaggata]'}")
        print(f"    median={np.median(c):.0f} ms   KDE peaks={npk}")
        if mod["ok"]:
            mu = mod["mu"]
            print(f"    GMM dBIC(1vs2)={mod['dbic']:+.1f}  (>0 favorisce 2 comp)")
            print(f"    GMM 2 medie: {mu[0]:.0f} / {mu[1]:.0f} ms  "
                  f"(distanza {mu[1]-mu[0]:.0f} ms)")
            print(f"    valle interna reale (criterio attuale): {mod['bimodal']}"
                  + (f"  @ {mod['valley']:.0f} ms" if mod["valley"] else ""))
        else:
            print("    GMM: dati insufficienti")
        # se i due modi GMM esistono, prova lo split morfologico a meta' strada
        if mod["ok"] and mod["mu"]:
            valley = mod["valley"] if mod["valley"] else float(np.mean(mod["mu"]))
            morph = D.coupling_focus_morph(s["ecg_path"], valley)
            if morph:
                print(f"    split @ {valley:.0f} ms -> n_lo={morph['n_lo']} n_hi={morph['n_hi']}"
                      f"  QRS template r={morph['corr']:.3f}  "
                      f"(w {morph['w_lo']:.0f}/{morph['w_hi']:.0f} ms)")
                print(f"      -> {'STESSO focolaio (r alto)' if morph['corr']>0.97 else 'morfologia DIVERSA, da guardare'}")

    print("\n  --> Nota: il criterio attuale flagga 'bimodale' solo se la mixture-2")
    print("      ha una VALLE INTERNA reale E dBIC>0. Se 14:59 ha KDE a 2 picchi ma")
    print("      il GMM non trova valle interna, e' 'spalla'/asimmetria, non due modi")
    print("      separati. I numeri qui sopra dicono in quale dei due casi siamo.")


def main():
    print("Carico le sessioni (stessa pipeline di dashboard.py)...")
    sessions = build_sessions()
    if not sessions:
        print("Nessuna sessione."); return
    q1_motif_by_session(sessions)
    q2_bimodality(sessions)


if __name__ == "__main__":
    main()
