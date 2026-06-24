"""
Clustering-robustness check for the single-population PVC result (report Appendix B).

The main report concludes that the detected PVCs form one continuous monomorphic
population rather than discrete morphological subtypes. Because K-means always returns
k clusters whether or not real structure exists, this script stress-tests that
conclusion on the SAME amplitude-normalised morphology matrix used by the continuum
check in host/dashboard.py:

  - K-means at k = 2, 3, 4 with silhouette scores  (low = no genuine clusters)
  - GaussianMixture BIC k = 1..4 + separation of the 2-component solution
  - DBSCAN  (one dense core + diffuse halo, vs two comparable dense groups?)
  - a direct profile of the low-density "skirt" (the sparse points in the PCA panel):
    QRS morphology and which sessions they come from.

Run from the repository root (needs the recordings in logs/):

    python3 host/analyze_pvc_clustering.py

Prints a text summary; makes no plots and writes nothing. The numbers quoted in
Appendix B of the report come from this script.
"""
import os
import sys
import glob
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
np.seterr(all="ignore")
import dashboard as D   # reuse load_session, constants, TG (main() is guarded)

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    sys.exit("scikit-learn is required: pip install scikit-learn")

TG, WIN, N = D.TG, D.WIN, D.N_SAMPLES


def build_matrix():
    """Rebuild the exact amplitude-normalised PVC matrix used by the continuum check,
    plus per-PVC metadata (session, QRS width, amplitude, coupling)."""
    ecg_files = sorted(glob.glob("logs/ecg_*.csv"))
    ecg_files = [f for f in ecg_files if os.path.getsize(f) > D.MIN_FILE_SIZE_MB * 1_000_000]
    ecg_files = [f for f in ecg_files if D.label_from_path(f)[1] not in D.SKIP_SESSIONS]

    norm_list, sess, width, amp, rrpre, dur = [], [], [], [], [], []
    for ecg_path in ecg_files:
        label, _ = D.label_from_path(ecg_path)
        data = D.load_session(ecg_path)
        if data is None:
            continue
        t_ecg, vf, peaks, _ = data
        if not peaks:
            continue
        raws, meta = [], []
        for i, p in enumerate(peaks):           # mirror collect_traces(kind='pvc') + keep meta
            if p["cls"] != "pvc":
                continue
            pt = p["t"]
            mask = (t_ecg >= pt - WIN / 2) & (t_ecg <= pt + WIN / 2)
            if mask.sum() < N * 0.9:
                continue
            raws.append(np.interp(TG, t_ecg[mask] - pt, vf[mask]))
            meta.append((p["w"], p["amp"], (pt - peaks[i - 1]["t"]) * 1000 if i else np.nan))
        raws = np.array(raws)
        if raws.shape[0] < D.MIN_PVC_COUNT:
            continue
        pmax = raws.max(axis=1, keepdims=True)
        pmax = np.where(pmax > 0.1, pmax, 1.0)
        norm_list.append(raws / pmax)
        for w, a, rr in meta:
            sess.append(label); width.append(w); amp.append(a); rrpre.append(rr)
        dur.append((label, float(t_ecg[-1] / 60)))

    all_norm = np.concatenate(norm_list, axis=0)
    X = all_norm - all_norm.mean(axis=1, keepdims=True)
    trough = -all_norm[:, (TG > 0.05) & (TG < 0.25)].min(axis=1)
    return (X, np.array(sess), np.array(width), np.array(amp),
            np.array(rrpre), trough, dict(dur))


def main():
    X, sess, width, amp, rrpre, trough, durmap = build_matrix()
    print(f"PVC morphology matrix: {X.shape[0]} beats x {X.shape[1]} samples, "
          f"{len(set(sess))} sessions\n")

    X2 = PCA(2, svd_solver="full").fit_transform(X)
    X10 = PCA(10, svd_solver="full").fit_transform(X)

    print("K-means partitions + silhouette (low = no genuine clusters):")
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), min(3000, len(X)), replace=False)
    for k in (2, 3, 4):
        lab = KMeans(k, random_state=42, n_init=10).fit_predict(X)
        sizes = " / ".join(f"{s:,}" for s in sorted(np.bincount(lab), reverse=True))
        print(f"  k={k}: sizes {sizes:<28}  silhouette {silhouette_score(X[idx], lab[idx]):.3f}")

    print("\nGaussianMixture BIC (lower = better):")
    bics = []
    for k in (1, 2, 3, 4):
        g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=0).fit(X10)
        bics.append(g.bic(X10)); print(f"  k={k}: BIC = {g.bic(X10):,.0f}")
    g2 = GaussianMixture(2, covariance_type="full", n_init=5, random_state=0).fit(X10)
    gap = np.linalg.norm(g2.means_[0] - g2.means_[1])
    spread = np.sqrt(np.mean([np.trace(c) for c in g2.covariances_]))
    print(f"  2-component separation: weights {g2.weights_[0]:.2f}/{g2.weights_[1]:.2f}, "
          f"mean-gap/spread = {gap/spread:.2f}  (<~2 = heavily overlapping)")

    print("\nDBSCAN on standardised PCA-2D:")
    Z = (X2 - X2.mean(0)) / X2.std(0)
    nn = NearestNeighbors(n_neighbors=10).fit(Z)
    dens = nn.kneighbors(Z)[0][:, -1]
    eps = float(np.percentile(np.sort(dens), 90))
    lab = DBSCAN(eps=eps, min_samples=20).fit_predict(Z)
    ncl = len(set(lab)) - (1 if -1 in lab else 0)
    big = max(np.sum(lab == c) for c in set(lab) if c != -1)
    print(f"  {ncl} dense cluster(s); largest = {big:,} ({100*big/len(X):.0f}%), "
          f"noise/halo = {np.mean(lab == -1)*100:.1f}%")

    print("\nLow-density 'skirt' (3% least-dense in PCA-2D) vs core:")
    skirt = dens >= np.percentile(dens, 97); core = ~skirt
    m = lambda a, k: np.nanmedian(a[k])
    print(f"  width {m(width,skirt):.0f}/{m(width,core):.0f} ms   "
          f"amp {m(amp,skirt):.2f}/{m(amp,core):.2f} V   "
          f"trough {m(trough,skirt):.2f}/{m(trough,core):.2f}   "
          f"coupling {m(rrpre,skirt):.0f}/{m(rrpre,core):.0f} ms  (skirt/core)")
    print("  skirt share by session:")
    rows = [(100*(( sess==s)&skirt).sum()/(sess==s).sum(), s,
             durmap.get(s, float('nan'))) for s in set(sess)]
    for frac, s, d in sorted(rows, reverse=True):
        print(f"    {s}: {frac:4.1f}%   ({d:.0f} min)")


if __name__ == "__main__":
    main()
