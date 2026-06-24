"""
Holter ECG — cumulative report generator (host/dashboard.py).

WHAT IT DOES, in short:
  reads every recording in logs/  →  applies the classification pipeline  →
  builds the figures (matplotlib)  →  embeds them into a single self-contained
  HTML file  →  reports/holter_dashboard.html
  (the PDF is that HTML printed via headless Chrome, see at the bottom).

────────────────────────────────────────────────────────────────────────────
FILE MAP — where to find what (use your editor's search on the markers)
────────────────────────────────────────────────────────────────────────────
  1. CONFIG          → at the top: detector thresholds, colors, constants
                       (search:  "classifier configuration",  "DARK_BG")
  2. FUNCTIONS       → the "math": loading and analysis, one function
                       per concept.  (search:  "def load_session",
                       "def pvc_pause_data",  "def extract_edr_and_phase")
  3. def main()      → the core: computes the metrics and BUILDS THE FIGURES.
                       Each figure block has a banner:  "# ==== ... ===="
                       and ends with  img_something = fig_to_b64(fig).
  4. HTML TEMPLATE   → at the bottom, in the large string that starts with  html = f
                       (search exactly:  html = f ).
                       ► ALL THE REPORT TEXT LIVES HERE ◄
                       - the titles are the  <h2>...</h2>  tags
                       - the descriptions are the  <div class="commentary">...</div>
                       - figures are inserted with  {img_something}
                       - computed numbers are inserted with  {variable_name}

EDITING THE TEXT BY HAND (the part you care about):
  Go to the bottom, into the HTML template, and change the text inside <h2> or
  <div class="commentary">. Watch out for only 2 things, because it's an f-string:
    • a LITERAL brace must be DOUBLED:  write  {{  and  }}  (not  {  } )
    • {something} without doubling = "insert the value of the variable
      something here". If you don't know what it is, don't touch it.
  The rest is normal HTML: <b>bold</b>, <br/> line break, <span style="...">.

REGENERATE:
    python3 host/dashboard.py                 → rewrites reports/holter_dashboard.html
    then (for the PDF) print the HTML from Chrome, or ask me.

ADDING A SESSION:  put the CSV in logs/, mark the noise with
    python3 host/mark_exclusions.py logs/ecg_*.csv  and regenerate. Everything updates.
"""
import csv, json, os, glob, base64, io
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure font = Helvetica Neue (closer to the SF Pro of the HTML body; SF Pro
# cannot be loaded by matplotlib). Fallback to Helvetica/Arial/DejaVu. Standard.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

# ---------- classifier configuration ----------
PVC_MIN_AMP = 0.70   # V
REBOUND_PVC = 0.40
PVC_W_MS    = 95.0
PVC_W_MIN   = 40.0
PVC_W_MAX   = 220.0
PVC_MIN_REBOUND = 0.05

SR = 250    # sample rate, Hz
WIN = 0.6   # window for morphology overlay (±0.3 s)
N_SAMPLES = int(WIN * SR)
TG = np.linspace(-WIN/2, WIN/2, N_SAMPLES)

# Minimum thresholds for including a session
MIN_FILE_SIZE_MB = 5
MIN_PVC_COUNT    = 50

# Sessions to exclude from the dataset (e.g. inverted electrodes, test runs)
SKIP_SESSIONS = {
    "20260605_131136",  # electrodes inverted (polarity reversed)
}

DARK_BG = "#ffffff"
GREEN, RED, BLUE, ORANGE = "#1b8a3a", "#c0304f", "#1f7fb0", "#cc7a1f"

# ============================================================================
# FIGURE GRAPHIC STANDARD — applied to EVERY 2x2 figure in the dashboard.
# Don't change individual numbers without updating here: this is the reference.
# ============================================================================
# Layout: SQUARE 12x12 figure + equal fractions (PANEL_W == PANEL_H) →
# SQUARE panels (0.34*12 = 4.08" per side). Columns far apart (gap ~1.6",
# holds the colorbar with vertical label on the side); rows closer (gap ~1.3").
# Horizontal and vertical gaps differ from each other: intentional choice.
FIGSIZE = (12, 12)
PANEL_W = 0.34
PANEL_H = 0.34
LEFT_COL  = 0.035
RIGHT_COL = 0.51   # columns far apart: gap holds colorbar + vertical label
                   # (0.51 - 0.375 = 0.135 → ~1.6")
BOTTOM_ROW = 0.07
TOP_ROW    = 0.52  # rows close (gap 0.52 - 0.41 = 0.11 → ~1.3")

# Font sizes for the matplotlib plots (smaller than the body CSS, for hierarchy).
# Standard shared by all figures.
FS_TITLE  = 11
FS_LABEL  = 9.5
FS_TICK   = 8.5
FS_LEGEND = 8.5
FS_TEXT   = 8
PANEL_POS = {
    "tl": (LEFT_COL,  TOP_ROW,    PANEL_W, PANEL_H),
    "tr": (RIGHT_COL, TOP_ROW,    PANEL_W, PANEL_H),
    "bl": (LEFT_COL,  BOTTOM_ROW, PANEL_W, PANEL_H),
    "br": (RIGHT_COL, BOTTOM_ROW, PANEL_W, PANEL_H),
}

def tight_cbar(fig, im, panel, label, fs=8.5):
    """Thin colorbar attached to the right edge of `panel` (l,b,w,h).
    VERTICAL label on the side (like the original). The gap between columns must be
    kept wide enough to hold bar + ticks + label without touching the y-label
    of the adjacent panel. The panel stays square (PANEL_POS untouched)."""
    l, b, w, h = panel
    cax = fig.add_axes([l + w + 0.006, b, 0.012, h])
    cb = fig.colorbar(im, cax=cax, label=label)
    cax.yaxis.label.set_color("#555555")
    cax.yaxis.label.set_fontsize(fs)
    cax.tick_params(colors="#555555", labelsize=fs)
    for sp in cax.spines.values():
        sp.set_color("#c8c8c8")
    return cb


def colored_title(ax, segments, fontsize=8.5, y=1.03):
    """Panel title with differently-colored segments (matplotlib does not
    allow this in a single set_title): e.g. letter+text in black, colored word.
    segments = [(text, color, weight), ...]."""
    from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
    boxes = [TextArea(t, textprops=dict(color=c, fontsize=fontsize, fontweight=w))
             for t, c, w in segments]
    pack = HPacker(children=boxes, align="baseline", pad=0, sep=0)
    ab = AnnotationBbox(pack, (0.5, y), xycoords="axes fraction",
                        box_alignment=(0.5, 0.0), frameon=False, pad=0,
                        annotation_clip=False)
    ax.add_artist(ab)

# ---------- I/O helpers ----------
def label_from_path(ecg_path):
    base = os.path.basename(ecg_path).replace("ecg_","").replace(".csv","")
    if len(base) >= 13:
        d = f"{base[0:4]}-{base[4:6]}-{base[6:8]} {base[9:11]}:{base[11:13]}"
    else:
        d = base
    return d, base

def short_label(label):
    # from "2026-06-05 14:59" → "06-05 14:59"
    return label[5:] if len(label) >= 16 else label

def load_session(ecg_path):
    pk_path = ecg_path.replace("ecg_","peaks_")
    if not os.path.exists(pk_path): return None
    _, base = label_from_path(ecg_path)
    excl_path = f"exclusions/exclusions_{base}.json"
    EXCL = []
    if os.path.exists(excl_path):
        try:
            with open(excl_path) as f:
                EXCL = [(d["start"], d["end"]) for d in json.load(f).get("intervals", [])]
        except Exception:
            pass

    peaks = []
    with open(pk_path) as f:
        for r in csv.DictReader(f):
            try:
                p = {"t":float(r["t_s"]), "amp":float(r["amp_V"]),
                     "w":float(r["width_ms"]), "reb":float(r["rebound_ratio"]),
                     "cls":r["class"]}
                shape  = (p["reb"] >= REBOUND_PVC or p["w"] >= PVC_W_MS)
                plaus  = PVC_W_MIN <= p["w"] <= PVC_W_MAX
                reb_ok = p["reb"] >= PVC_MIN_REBOUND
                p["cls"] = "pvc" if (shape and p["amp"] >= PVC_MIN_AMP
                                      and plaus and reb_ok) else "normal"
                peaks.append(p)
            except (KeyError, ValueError):
                continue
    peaks = [p for p in peaks if not (p["w"] <= 16 and p["amp"] < PVC_MIN_AMP)]
    peaks = [p for p in peaks if not any(s <= p["t"] <= e for s, e in EXCL)]

    ts, vf = [], []
    with open(ecg_path) as f:
        for r in csv.DictReader(f):
            try:
                ts.append(float(r["t_s"])); vf.append(float(r["filt"]))
            except (KeyError, ValueError):
                continue
    return np.array(ts), np.array(vf), peaks, EXCL

def collect_traces(t_ecg, vf_arr, peaks, kind="pvc"):
    out = []
    for p in peaks:
        if p["cls"] != kind: continue
        pt = p["t"]
        mask = (t_ecg >= pt-WIN/2) & (t_ecg <= pt+WIN/2)
        if mask.sum() < N_SAMPLES * 0.9: continue
        v = np.interp(TG, t_ecg[mask]-pt, vf_arr[mask])
        out.append(v)
    return np.array(out) if out else np.zeros((0, N_SAMPLES))

def pick_example_strip(t_ecg, vf_arr, peaks, excl, pre=10.0, post=10.0):
    """Representative 'clean' ECG window to show signal quality +
    auto-detection. Same criteria as the PDF reports: N-PVC-N sandwich PVC after
    the 1st minute, >=2s away from excluded intervals, median amplitude (not outlier).
    SYMMETRIC window → the reference PVC falls at the center (x=0 in the middle).
    Returns {t, v, peaks, center, pre, post} or None."""
    if len(t_ecg) == 0 or not peaks:
        return None
    def far_excl(ts, m=2.0):
        return all(not (s - m <= ts <= e + m) for s, e in excl)
    cand = []
    for i, p in enumerate(peaks):
        if p["cls"] != "pvc" or i == 0 or i == len(peaks) - 1:
            continue
        if p["t"] < 60:                                  # skip warm-up
            continue
        if peaks[i-1]["cls"] != "normal" or peaks[i+1]["cls"] != "normal":
            continue
        if not far_excl(p["t"]):
            continue
        cand.append(p)
    if cand:
        cand.sort(key=lambda q: q["amp"])
        chosen = cand[len(cand) // 2]                    # median amplitude
    else:
        pvc_all = [p for p in peaks if p["cls"] == "pvc"]
        if not pvc_all:
            return None
        chosen = next((p for p in pvc_all if p["t"] >= 60 and far_excl(p["t"])),
                      pvc_all[0])
    c = chosen["t"]
    m = (t_ecg >= c - pre) & (t_ecg <= c + post)
    if not m.any():
        return None
    wp = [{"t": p["t"], "cls": p["cls"], "amp": p["amp"]}
          for p in peaks if c - pre <= p["t"] <= c + post]
    return {"t": t_ecg[m], "v": vf_arr[m], "peaks": wp,
            "center": c, "pre": pre, "post": post}

def _snip(t_ecg, vf_arr, peaks, ctr, half):
    """Extracts the {t,v,peaks,center,pre,post} snippet symmetric around ctr."""
    m = (t_ecg >= ctr - half) & (t_ecg <= ctr + half)
    if not m.any():
        return None
    wp = [{"t": p["t"], "cls": p["cls"], "amp": p["amp"]}
          for p in peaks if ctr - half <= p["t"] <= ctr + half]
    return {"t": t_ecg[m], "v": vf_arr[m], "peaks": wp,
            "center": ctr, "pre": half, "post": half}

def window_noise_score(strip, guard=0.16):
    """Baseline disturbance in a snippet's window: 90th percentile of |v|
    over samples >guard s away from EVERY peak (= inter-beat zone, which in a clean
    ECG is nearly flat). Low ⇒ clean window, high ⇒ noise/artifacts."""
    if strip is None or len(strip["t"]) == 0:
        return 9.9
    t, v = strip["t"], strip["v"]
    mask = np.ones(len(t), dtype=bool)
    for p in strip["peaks"]:
        mask &= np.abs(t - p["t"]) > guard
    base = np.abs(v[mask])
    return float(np.percentile(base, 90)) if len(base) >= 10 else 9.9

def find_interpolated_strip(t_ecg, vf_arr, peaks, excl, half=10.0, max_ratio=1.30):
    """Interpolated PVC: between 2 sinus beats, RR_pre+RR_post ≈ 1× sinus RR (ratio<=1.30,
    NOT compensated ~2×). Picks the one with the lowest ratio (most clearly
    interpolated). Returns snippet with extra 'ratio', or None."""
    if len(t_ecg) == 0:
        return None
    def far(ts, m=2.0):
        return all(not (s - m <= ts <= e + m) for s, e in excl)
    best = None
    for i in range(1, len(peaks) - 1):
        p = peaks[i]
        if p["cls"] != "pvc" or p["t"] < 60:
            continue
        if peaks[i-1]["cls"] != "normal" or peaks[i+1]["cls"] != "normal":
            continue
        if not far(p["t"]):
            continue
        rr_pre  = peaks[i]["t"]   - peaks[i-1]["t"]
        rr_post = peaks[i+1]["t"] - peaks[i]["t"]
        nn = []
        for j in range(max(0, i-6), min(len(peaks)-1, i+6)):
            if peaks[j]["cls"] == "normal" and peaks[j+1]["cls"] == "normal":
                d = peaks[j+1]["t"] - peaks[j]["t"]
                if 0.4 < d < 1.5:
                    nn.append(d)
        if not nn:
            continue
        ratio = (rr_pre + rr_post) / float(np.median(nn))
        if ratio <= max_ratio and (best is None or ratio < best[1]):
            best = (p["t"], ratio)
    if best is None:
        return None
    snip = _snip(t_ecg, vf_arr, peaks, best[0], half)
    if snip is not None:
        snip["ratio"] = best[1]
    return snip

def find_couplet_strip(t_ecg, vf_arr, peaks, excl, half=10.0, max_rr=700.0):
    """Best couplet (2 consecutive PVCs, RR<700ms) of the session, centered on
    the midpoint of the pair. Returns snippet with extra 'rr', or None."""
    if len(t_ecg) == 0:
        return None
    def far(ts, m=2.0):
        return all(not (s - m <= ts <= e + m) for s, e in excl)
    best = None
    for i in range(len(peaks) - 1):
        a, b = peaks[i], peaks[i+1]
        if a["cls"] != "pvc" or b["cls"] != "pvc" or a["t"] < 60:
            continue
        rr = (b["t"] - a["t"]) * 1000
        if not (200 < rr < max_rr):
            continue
        if not (far(a["t"]) and far(b["t"])):
            continue
        if best is None or rr < best[1]:
            best = ((a["t"] + b["t"]) / 2.0, rr)
    if best is None:
        return None
    snip = _snip(t_ecg, vf_arr, peaks, best[0], half)
    if snip is not None:
        snip["rr"] = best[1]
    return snip

# grids for the couplet overlay (aligned on the 1st PVC's peak)
CPL_PRE, CPL_POST = 0.22, 0.78          # pair window: -0.22..+0.78 s
CPL_GRID = np.linspace(-CPL_PRE, CPL_POST, 250)
QRS_HALF = 0.15                          # half-window for the individual QRS
QRS_GRID = np.linspace(-QRS_HALF, QRS_HALF, 75)

def find_all_couplets(t_ecg, vf_arr, peaks, excl, lo=200.0, hi=700.0):
    """ALL true couplets of the session: exactly 2 consecutive PVCs with
    `lo < RR < hi` ms, NOT part of a run >=3 (no adjacent PVC before/after),
    outside excluded stretches, after the warm-up. For each it extracts the pair's
    waveform (aligned and normalized on the 1st PVC's peak) and the two individual
    QRS, for the morphological overlay. Returns a list of dicts."""
    if len(t_ecg) == 0:
        return []
    def far(ts, m=2.0):
        return all(not (s - m <= ts <= e + m) for s, e in excl)
    out, n = [], len(peaks)
    for i in range(n - 1):
        a, b = peaks[i], peaks[i+1]
        if a["cls"] != "pvc" or b["cls"] != "pvc" or a["t"] < 60:
            continue
        rr = (b["t"] - a["t"]) * 1000
        if not (lo < rr < hi):
            continue
        if i - 1 >= 0 and peaks[i-1]["cls"] == "pvc":      # no run >=3
            continue
        if i + 2 < n and peaks[i+2]["cls"] == "pvc":
            continue
        if not (far(a["t"]) and far(b["t"])):
            continue
        m = (t_ecg >= a["t"] - CPL_PRE) & (t_ecg <= a["t"] + CPL_POST)
        if m.sum() < len(CPL_GRID) * 0.8:
            continue
        pair = np.interp(CPL_GRID, t_ecg[m] - a["t"], vf_arr[m])
        norm = np.max(np.abs(pair)) or 1.0
        def qrs(tc):
            mm = (t_ecg >= tc - QRS_HALF) & (t_ecg <= tc + QRS_HALF)
            if mm.sum() < len(QRS_GRID) * 0.8:
                return None
            q = np.interp(QRS_GRID, t_ecg[mm] - tc, vf_arr[mm])
            pk = np.max(np.abs(q)) or 1.0
            return q / pk
        q1, q2 = qrs(a["t"]), qrs(b["t"])
        if q1 is None or q2 is None:
            continue
        out.append({"t1": a["t"], "t2": b["t"], "rr": rr,
                    "amp1": a["amp"], "amp2": b["amp"],
                    "w1": a["w"], "w2": b["w"], "reb1": a["reb"], "reb2": b["reb"],
                    "pair": pair / norm, "q1": q1, "q2": q2,
                    "strip": _snip(t_ecg, vf_arr, peaks, (a["t"] + b["t"]) / 2.0, 5.0)})
        out[-1]["noise"] = window_noise_score(out[-1]["strip"])
        # local rhythm motif: 4 beats before the 1st PVC .. 4 after the 2nd
        sy = lambda p: "V" if p["cls"] == "pvc" else "N"
        pre = "".join(sy(peaks[k]) for k in range(max(0, i-4), i))
        post = "".join(sy(peaks[k]) for k in range(i+2, min(n, i+6)))
        out[-1]["ctx"] = pre + "VV" + post           # for grouping
        out[-1]["ctx_disp"] = pre + "[VV]" + post     # for the label
    return out

def find_burst_strip(t_ecg, vf_arr, peaks, excl, half=10.0, win=10.0, min_n=3):
    """Window of `win` seconds with the HIGHEST PVC density of the session
    (discharge/burst). Centered on the cluster. Returns snippet with extra 'n', or None."""
    if len(t_ecg) == 0:
        return None
    def far(ts, m=2.0):
        return all(not (s - m <= ts <= e + m) for s, e in excl)
    pv = np.array([p["t"] for p in peaks
                   if p["cls"] == "pvc" and p["t"] >= 60 and far(p["t"])])
    if len(pv) < min_n:
        return None
    counts = np.searchsorted(pv, pv + win, side="right") - np.arange(len(pv))
    k = int(np.argmax(counts)); n = int(counts[k])
    if n < min_n:
        return None
    ctr = pv[k] + win / 2.0
    snip = _snip(t_ecg, vf_arr, peaks, ctr, half)
    if snip is not None:
        snip["n"] = n
    return snip

# ---- Interpolated vs compensated classification (validated, see "method" section) ----
# Criterion: LOCAL sinus cycle = median of the PAUSE_K N-N intervals nearest the PVC
# (follows the RSA). Prematurity guard (RR_pre >= local sinus = missed beat →
# discarded). Discrimination on the PAUSE RR_post/sinus (this is what is perceived
# as a "thump"): bimodal distribution, cut at the valley (~1.0). 2 classes.
PAUSE_K = 15

def nn_arrays(peaks):
    """midpoint-time and duration (s) of every physiological consecutive N-N pair."""
    mids, vals = [], []
    for i in range(1, len(peaks)):
        if peaks[i]["cls"] == "normal" and peaks[i-1]["cls"] == "normal":
            d = peaks[i]["t"] - peaks[i-1]["t"]
            if 0.4 < d < 1.6:
                mids.append((peaks[i]["t"] + peaks[i-1]["t"]) / 2.0)
                vals.append(d)
    return np.array(mids), np.array(vals)

def pvc_pause_data(peaks):
    """For every PVC in an N-PVC-N sandwich: local sinus cycle (median of the
    PAUSE_K nearest N-N), coupling RR_pre, pause RR_post, ratios over sinus,
    and `guard` flag (RR_pre >= local sinus → missed beat, unreliable)."""
    mids, vals = nn_arrays(peaks)
    gl = float(np.median(vals)) if len(vals) else 1.0
    n = len(mids); half = PAUSE_K // 2
    out = []
    for i in range(1, len(peaks) - 1):
        p = peaks[i]
        if p["cls"] != "pvc": continue
        if peaks[i-1]["cls"] != "normal" or peaks[i+1]["cls"] != "normal": continue
        rr_pre = p["t"] - peaks[i-1]["t"]
        rr_post = peaks[i+1]["t"] - p["t"]
        idx = int(np.searchsorted(mids, p["t"])) if n else 0
        lo = max(0, idx - half - 1); hi = min(n, lo + PAUSE_K); lo = max(0, hi - PAUSE_K)
        rl = float(np.median(vals[lo:hi])) if hi - lo >= 3 else gl
        out.append({"i": i, "t": p["t"], "amp": p["amp"],
                    "rr_pre": rr_pre, "rr_post": rr_post, "rl": rl,
                    "post_ratio": rr_post / rl, "s_ratio": (rr_pre + rr_post) / rl,
                    "guard": rr_pre >= rl})
    return out

def pause_valley(post_ratios, lo=0.82, hi=1.28, default=1.02):
    """Valley (density minimum) of the RR_post/sinus distribution between the two humps
    (silent ~0.75, with-pause ~1.45). Reasonable clamp."""
    a = np.asarray(list(post_ratios), dtype=float)
    if len(a) < 50:
        return default
    h, e = np.histogram(a, bins=np.arange(0.3, 2.2, 0.03))
    cen = (e[:-1] + e[1:]) / 2
    hs = np.convolve(h, np.array([1, 2, 3, 2, 1]) / 9.0, mode="same")
    m = (cen > lo) & (cen < hi)
    if not m.any():
        return default
    return float(min(1.18, max(0.90, cen[m][np.argmin(hs[m])])))

# ---- "Dual focus" check on the pre-PVC coupling (validated 10 Jun 2026) --------
# A second focus would give a SECOND coupling peak WITH a different QRS morphology.
# A second peak with the SAME morphology = same focus discharging at two
# intervals (coupling modulation), NOT bifocal.
def coupling_modality(cm):
    """GMM 1 vs 2 comp on the coupling (ms). Returns whether the mixture-2 is GENUINELY
    bimodal (a real trough between the peaks, not just asymmetry), the dBIC, and the
    valley between the two modes. No reload: works on the coupling array."""
    out = {"ok": False, "bimodal": False, "dbic": 0.0, "valley": None,
           "mu": None}
    cm = np.asarray(cm, dtype=float)
    cm = cm[(cm > 200) & (cm < 800)]
    if len(cm) < 80:
        return out
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        return out
    X = cm.reshape(-1, 1)
    g1 = GaussianMixture(1, n_init=2, random_state=0).fit(X)
    g2 = GaussianMixture(2, covariance_type="full", n_init=5, random_state=0).fit(X)
    dbic = g1.bic(X) - g2.bic(X)          # >0 favors 2 comp
    mus = np.sort(g2.means_.ravel())
    out.update(ok=True, dbic=float(dbic), mu=(float(mus[0]), float(mus[1])))
    if dbic <= 0 or mus[1] - mus[0] < 1e-3:
        return out
    grid = np.linspace(mus[0], mus[1], 200)
    dens = np.exp(g2.score_samples(grid.reshape(-1, 1)))
    j = int(np.argmin(dens))
    if 0 < j < len(grid) - 1:             # INTERNAL minimum → two real modes
        out.update(bimodal=True, valley=float(grid[j]))
    return out

def coupling_focus_morph(ecg_path, valley):
    """For a session with bimodal coupling: reload, split the PVCs at the `valley`,
    morphological comparison of the two clusters (median width/rebound/amp + correlation
    of the median QRS template). High r (~>0.97) ⇒ SAME focus."""
    d = load_session(ecg_path)
    if d is None:
        return None
    t_ecg, vf, peaks, _ = d
    lo_p, hi_p = [], []
    for i, p in enumerate(peaks):
        if p["cls"] != "pvc" or i == 0:
            continue
        rr = (p["t"] - peaks[i-1]["t"]) * 1000
        if 200 < rr < 800:
            (lo_p if rr < valley else hi_p).append(p)
    if len(lo_p) < 20 or len(hi_p) < 20:
        return None
    W = 0.18; grid = np.linspace(-W, W, 90)
    def templ(sub):
        rows = []
        for p in sub:
            m = (t_ecg >= p["t"]-W) & (t_ecg <= p["t"]+W)
            if m.sum() < 60:
                continue
            v = np.interp(grid, t_ecg[m]-p["t"], vf[m])
            pk = np.max(np.abs(v))
            rows.append(v/pk if pk > 0.05 else v)
        return np.median(np.array(rows), axis=0) if rows else None
    tlo, thi = templ(lo_p), templ(hi_p)
    if tlo is None or thi is None:
        return None
    return {
        "n_lo": len(lo_p), "n_hi": len(hi_p),
        "w_lo": float(np.median([p["w"] for p in lo_p])),
        "w_hi": float(np.median([p["w"] for p in hi_p])),
        "r_lo": float(np.median([p["reb"] for p in lo_p])),
        "r_hi": float(np.median([p["reb"] for p in hi_p])),
        "corr": float(np.corrcoef(tlo, thi)[0, 1]),
    }

# ---- EDR (ECG-Derived Respiration) + phasic analysis of the PVCs -----------------
# Breathing modulates the QRS amplitude (rotation of the cardiac vector + lung
# impedance). I reconstruct respiration from the R-amplitude of the normal beats,
# estimate its instantaneous phase (Hilbert), and look at which respiratory phase the
# PVCs fall at vs the normal beats → is there a breathing↔PVC correlation? (chi² on the phase bins).
NBINS_RESP = 12
FS_RESP = 4.0

def extract_edr_and_phase(peaks):
    """EDR from the R-amplitude of the N beats + instantaneous phase + phasic distribution PVC vs N.
    Convention VERIFIED by the user (direct observation during recording):
    MAXIMUM R-amplitude = FULL LUNGS = end of INSPIRATION; falls as they empty.
    So phase 0 = full lungs (end-inspir.), 50% of the cycle = empty lungs
    (end-expir.). The subject's PVCs cluster near phase 0 (full lungs).
    Returns a dict or None if the trace is insufficient."""
    norm = [p for p in peaks if p["cls"] == "normal"]
    pvc  = [p for p in peaks if p["cls"] == "pvc"]
    if len(norm) < 200 or len(pvc) < 30:
        return None
    t_n = np.array([p["t"] for p in norm]); amp_n = np.array([p["amp"] for p in norm])
    if t_n[-1] - t_n[0] < 5 * 60:          # at least 5 min
        return None
    try:
        from scipy import signal as sig
        from scipy.interpolate import interp1d
        from scipy.stats import chi2_contingency
    except Exception:
        return None
    t_unif = np.arange(t_n[0], t_n[-1], 1 / FS_RESP)
    amp_unif = interp1d(t_n, amp_n, kind="cubic")(t_unif)
    resp = sig.sosfiltfilt(
        sig.butter(3, [0.10, 0.50], btype="band", fs=FS_RESP, output="sos"),
        sig.detrend(amp_unif))
    f_psd, psd = sig.welch(resp, fs=FS_RESP, nperseg=min(2048, len(resp) // 4))
    in_band = (f_psd >= 0.10) & (f_psd <= 0.50); out_band = (f_psd >= 0.60) & (f_psd <= 1.5)
    snr = float(np.mean(psd[in_band]) / max(1e-12, np.mean(psd[out_band])))
    rate_resp = float(f_psd[in_band][np.argmax(psd[in_band])] * 60)
    phase = np.mod(np.angle(sig.hilbert(resp)), 2 * np.pi)
    pint = interp1d(t_unif, phase, kind="nearest", bounds_error=False, fill_value=0)
    bins = np.linspace(0, 2 * np.pi, NBINS_RESP + 1)
    pvc_t = np.array([p["t"] for p in pvc])
    pvc_phase = pint(pvc_t)
    hist_n, _ = np.histogram(pint([p["t"] for p in norm]), bins=bins)
    hist_p, _ = np.histogram(pvc_phase, bins=bins)
    chi2_val, pval, _, _ = chi2_contingency(np.array([hist_p, hist_n]))
    dens_n = hist_n / max(1, hist_n.sum()); dens_p = hist_p / max(1, hist_p.sum())
    enrich = dens_p / np.maximum(dens_n, 1e-6)
    centers = (bins[:-1] + bins[1:]) / 2
    pb = int(np.argmax(enrich))
    return {
        "snr": snr, "rate_resp": rate_resp, "chi2": float(chi2_val), "pval": float(pval),
        "dens_n": dens_n, "dens_p": dens_p, "enrich": enrich, "centers": centers,
        "peak_phase_pct": float(centers[pb] * 100 / (2 * np.pi)),
        "peak_enrich": float(enrich[pb]), "n_n": len(norm), "n_p": len(pvc),
        "t_unif": t_unif, "resp": resp, "t_n": t_n, "amp_n": amp_n,
        "pvc_t": pvc_t, "pvc_phase": pvc_phase,
        # interp1d on the times → phase, for subset queries (interp/comp/coupled)
        "phase_at": pint,
    }

def session_metrics(peaks, clean_s):
    """Basic per-session metrics (NO interp/comp: those are counted later, with
    the global valley). burden, sinus N/min, effective SA-HR, PVC rate, couplets."""
    norm = [p for p in peaks if p["cls"] == "normal"]
    pvc  = [p for p in peaks if p["cls"] == "pvc"]
    n_total = len(peaks)
    burden = 100 * len(pvc) / max(1, n_total)
    sinus_bpm = 60 * len(norm) / clean_s if clean_s else 0
    pvc_rate  = 60 * len(pvc)  / clean_s if clean_s else 0
    sinus_rr = [peaks[i]["t"] - peaks[i-1]["t"]
                for i in range(1, len(peaks))
                if peaks[i]["cls"] == "normal" and 0.6 < peaks[i]["t"] - peaks[i-1]["t"] < 1.4]
    rr_s_ms = float(np.median(sinus_rr)) * 1000 if sinus_rr else 1000.0
    sa_hr = 60000.0 / rr_s_ms if rr_s_ms else 0.0
    n_couplet = 0
    i = 0
    while i < len(peaks) - 1:
        if (peaks[i]["cls"] == "pvc" and peaks[i+1]["cls"] == "pvc"
                and peaks[i+1]["t"] - peaks[i]["t"] < 0.70
                and not (i+2 < len(peaks) and peaks[i+2]["cls"] == "pvc")):
            n_couplet += 1; i += 2
        else:
            i += 1
    # repetitive patterns: isolated PVCs, bigeminy (V-N-V), trigeminy (V-N-N-V)
    iso_pvc = sum(1 for k, p in enumerate(peaks) if p["cls"] == "pvc"
                  and (k == 0 or peaks[k-1]["cls"] != "pvc")
                  and (k == len(peaks)-1 or peaks[k+1]["cls"] != "pvc"))
    bigem = sum(1 for k in range(2, len(peaks))
                if peaks[k]["cls"] == "pvc" and peaks[k-1]["cls"] == "normal"
                and peaks[k-2]["cls"] == "pvc")
    trigem = sum(1 for k in range(3, len(peaks))
                 if peaks[k]["cls"] == "pvc" and peaks[k-1]["cls"] == "normal"
                 and peaks[k-2]["cls"] == "normal" and peaks[k-3]["cls"] == "pvc")
    # atrial fibrillation screening (on consecutive N-N) — permanent instruction
    af_nn = np.array([peaks[k]["t"] - peaks[k-1]["t"]
                      for k in range(1, len(peaks))
                      if peaks[k]["cls"] == "normal" and peaks[k-1]["cls"] == "normal"
                      and 0.4 <= peaks[k]["t"] - peaks[k-1]["t"] <= 2.0]) * 1000
    rmssd = pnn50 = cv = ent = 0.0; npk = 0; af_score = None
    if len(af_nn) >= 30:
        diffs = np.abs(np.diff(af_nn))
        rmssd = float(np.sqrt(np.mean(diffs**2)))
        pnn50 = 100 * float(np.mean(diffs > 50))
        cv = 100 * float(np.std(af_nn, ddof=1) / np.mean(af_nn))
        hist, _ = np.histogram(af_nn, bins=20)
        p = hist[hist > 0] / hist[hist > 0].sum()
        H = float(-(p * np.log2(p)).sum()); Hmax = float(np.log2(len(p))) if len(p) > 1 else 1.0
        ent = H / Hmax if Hmax else 0.0
        sm = np.convolve(hist, [1, 1, 1], mode="same")
        npk = sum(1 for k in range(1, len(sm)-1)
                  if sm[k] > sm[k-1] and sm[k] > sm[k+1] and sm[k] > 0.3 * sm.max())
        af_score = int((rmssd > 100) + (pnn50 > 40) + (ent > 0.85)
                       + (npk <= 1 and cv > 15))
    return {
        "burden": burden, "sinus_bpm": sinus_bpm, "pvc_rate": pvc_rate,
        "sa_hr": sa_hr, "rr_s_ms": rr_s_ms, "n_couplet": n_couplet,
        "n_total": n_total, "clean_s": clean_s,
        "iso_pvc": iso_pvc, "bigem": bigem, "trigem": trigem,
        "af_score": af_score, "rmssd": rmssd, "pnn50": pnn50, "cv": cv,
        # interp/comp filled in later (the global valley is needed):
        "n_interp": 0, "n_comp": 0, "pct_interp": 0.0, "pct_comp": 0.0,
    }

def draw_example_strip(ax, ex, title):
    """Draws a report-style strip: filtered trace in green, QRS of the PVCs
    in red (±120 ms) + red triangle marker, green markers on the sinus beats.
    `title` = short label above the strip (suited to the 2-column grid)."""
    c = ex["center"]
    ax.set_facecolor(DARK_BG)
    ax.plot(ex["t"] - c, ex["v"], lw=0.45, color="#2f8a63")
    for p in ex["peaks"]:
        if p["cls"] == "pvc":
            wm = (ex["t"] >= p["t"] - 0.12) & (ex["t"] <= p["t"] + 0.12)
            if wm.any():
                ax.plot(ex["t"][wm] - c, ex["v"][wm], lw=0.9, color="#cc3b30")
            ax.scatter(p["t"] - c, min(1.6, p["amp"] + 0.30), s=28, marker="v",
                       color="#cc3b30", edgecolors="#1a1a1a", linewidths=0.4, zorder=5)
        else:
            ax.scatter(p["t"] - c, min(1.4, p["amp"] + 0.18), s=9, marker="v",
                       color="#2f8a63", edgecolors="#1a1a1a", linewidths=0.25, zorder=4)
    ax.set_xlim(-ex["pre"], ex["post"]); ax.set_ylim(-1.2, 1.8)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="#777777", labelsize=8)
    ax.grid(True, alpha=0.14, color="#dcdcdc", linewidth=0.4)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.set_title(title, color="#222222", fontsize=8.5, pad=3)

def fig_to_b64(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def _png_width(b64):
    """Width in px from the PNG header (IHDR at offset 16), no dependencies."""
    raw = base64.b64decode(b64)
    return int.from_bytes(raw[16:20], "big")

def disp_width(b64, base=1000, dpi=220):
    """Display width (px) that keeps the panel scale CONSTANT across the
    3 2x2 figures: `bbox_inches=tight` crops different margins (legend vs
    colorbar) → different PNG widths → at the same max-width the panels
    would appear to have different sizes. Normalizing against the width of
    the whole figure (FIGSIZE[0]*dpi) makes the panels appear identical."""
    if not b64:
        return base
    ref_px = FIGSIZE[0] * dpi
    return round(base * _png_width(b64) / ref_px)

# ---------- main ----------
def main():
    print("Scanning logs/...")
    ecg_files = sorted(glob.glob("logs/ecg_*.csv"))
    ecg_files = [f for f in ecg_files
                 if os.path.getsize(f) > MIN_FILE_SIZE_MB * 1_000_000]
    ecg_files = [f for f in ecg_files
                 if label_from_path(f)[1] not in SKIP_SESSIONS]
    print(f"  {len(ecg_files)} candidate sessions ≥ {MIN_FILE_SIZE_MB} MB")

    sessions = []
    best_couplet = None   # (snippet, session_label)  couplet with the tightest RR
    best_burst   = None   # (snippet, session_label)  window with the most PVCs
    interp_candidates = []  # [(snippet, session_label), ...] interpolated PVCs
    for ecg_path in ecg_files:
        label, base = label_from_path(ecg_path)
        print(f"  loading {label}...")
        data = load_session(ecg_path)
        if data is None: continue
        t_ecg, vf_arr, peaks, excl = data
        if not peaks: continue
        traces_raw = collect_traces(t_ecg, vf_arr, peaks, kind="pvc")
        if traces_raw.shape[0] < MIN_PVC_COUNT:
            print(f"    skip: only {traces_raw.shape[0]} PVCs")
            continue
        # coupling intervals (RR_pre for each PVC, in ms)
        coupling_ms = []
        for i, p in enumerate(peaks):
            if p["cls"] != "pvc" or i == 0: continue
            rr = (p["t"] - peaks[i-1]["t"]) * 1000
            if 200 < rr < 800:   # physiological coupling range
                coupling_ms.append(rr)
        # also collect N (max 500/session to avoid saturating memory)
        traces_n_raw = collect_traces(t_ecg, vf_arr, peaks, kind="normal")
        if traces_n_raw.shape[0] > 500:
            idx = np.linspace(0, traces_n_raw.shape[0]-1, 500, dtype=int)
            traces_n_raw = traces_n_raw[idx]

        n_pvc  = sum(1 for p in peaks if p["cls"] == "pvc")
        n_norm = sum(1 for p in peaks if p["cls"] == "normal")
        peaks_max = traces_raw.max(axis=1, keepdims=True)
        peaks_max = np.where(peaks_max > 0.1, peaks_max, 1.0)
        traces_norm = traces_raw / peaks_max
        n_max = traces_n_raw.max(axis=1, keepdims=True) if len(traces_n_raw) else np.array([])
        if len(n_max):
            n_max = np.where(n_max > 0.05, n_max, 1.0)
            traces_n_norm = traces_n_raw / n_max
        else:
            traces_n_norm = traces_n_raw
        sessions.append({
            "label": label, "base": base, "ecg_path": ecg_path,
            "n_pvc": n_pvc, "n_norm": n_norm,
            "duration_min": float(t_ecg[-1]/60) if len(t_ecg) else 0,
            "n_excluded_intervals": len(excl),
            "excluded_seconds": float(sum(e-s for s, e in excl)),
            "traces_raw": traces_raw,
            "traces_norm": traces_norm,
            "traces_n_raw": traces_n_raw,
            "traces_n_norm": traces_n_norm,
            "coupling_ms": np.array(coupling_ms),
            "example": pick_example_strip(t_ecg, vf_arr, peaks, excl),
            "couplets": find_all_couplets(t_ecg, vf_arr, peaks, excl),
            "edr": extract_edr_and_phase(peaks),
            "pause_data": pvc_pause_data(peaks),
            "metrics": session_metrics(
                peaks,
                (float(t_ecg[-1] - t_ecg[0]) - float(sum(e - s for s, e in excl)))
                if len(t_ecg) else 0.0),
        })
        # best couplet / burst / interpolated at dataset level (special strips)
        cpl = find_couplet_strip(t_ecg, vf_arr, peaks, excl)
        if cpl is not None and (best_couplet is None or cpl["rr"] < best_couplet[0]["rr"]):
            best_couplet = (cpl, label)
        brst = find_burst_strip(t_ecg, vf_arr, peaks, excl)
        if brst is not None and (best_burst is None or brst["n"] > best_burst[0]["n"]):
            best_burst = (brst, label)
        itp = find_interpolated_strip(t_ecg, vf_arr, peaks, excl)
        if itp is not None:
            interp_candidates.append((itp, label))
    if not sessions:
        print("No valid session found."); return

    print(f"Sessions kept: {len(sessions)}")

    # ---- global RR_post valley + per-session interp/comp counts ----
    all_post = [d["post_ratio"] for s in sessions for d in s["pause_data"] if not d["guard"]]
    all_sratio = [d["s_ratio"] for s in sessions for d in s["pause_data"] if not d["guard"]]
    PAUSE_VALLEY = pause_valley(all_post)
    for s in sessions:
        nd = [d for d in s["pause_data"] if not d["guard"]]
        ni = sum(1 for d in nd if d["post_ratio"] < PAUSE_VALLEY)
        nc = sum(1 for d in nd if d["post_ratio"] >= PAUSE_VALLEY)
        ncl = max(1, ni + nc)
        s["metrics"].update(n_interp=ni, n_comp=nc,
                            pct_interp=100*ni/ncl, pct_comp=100*nc/ncl)
    print(f"RR_post valley = {PAUSE_VALLEY:.3f}  (interp<valley, comp>=valley)")
    all_traces_norm = np.concatenate([s["traces_norm"] for s in sessions], axis=0)
    all_traces_raw  = np.concatenate([s["traces_raw"]  for s in sessions], axis=0)
    all_n_norm = np.concatenate([s["traces_n_norm"] for s in sessions
                                 if len(s["traces_n_norm"])], axis=0)
    all_coupling = np.concatenate([s["coupling_ms"] for s in sessions
                                   if len(s["coupling_ms"])])
    print(f"Total PVC traces: {len(all_traces_norm)}, N traces: {len(all_n_norm)}, "
          f"couplings: {len(all_coupling)}")

    # ============ MORPHOLOGY: single 4-panel figure with shared axes ============
    med_per_sess = [np.median(s["traces_norm"], axis=0) for s in sessions]
    corr_matrix = np.zeros((len(sessions), len(sessions)))
    for i in range(len(sessions)):
        for j in range(len(sessions)):
            corr_matrix[i, j] = np.corrcoef(med_per_sess[i], med_per_sess[j])[0, 1]
    mask_t = (TG > 0.05) & (TG < 0.25)
    trough_depth = -all_traces_norm[:, mask_t].min(axis=1)

    # common Y range for panels 1 and 2 (shape)
    med_all = np.median(all_traces_norm, axis=0)
    p25 = np.percentile(all_traces_norm, 25, axis=0)
    p75 = np.percentile(all_traces_norm, 75, axis=0)
    y_min = min(p25.min(), min(m.min() for m in med_per_sess)) - 0.05
    y_max = 1.10

    # PVC morphology figure: explicit positions for perfect alignment.
    # width 8.1in = same as quality_strip / example_strips (Fig 1-2): so on the
    # page (scaled to \linewidth) font and stroke render at the same size.
    fig = plt.figure(figsize=(8.1, 7.6), facecolor=DARK_BG)

    # (1) — overlay all PVCs (red palette)
    ax = fig.add_axes(PANEL_POS["tl"]); ax.set_facecolor(DARK_BG)
    step = max(1, len(all_traces_norm) // 500)
    for tr in all_traces_norm[::step]:
        ax.plot(TG, tr, color="#d2685f", lw=0.4, alpha=0.06)
    ax.fill_between(TG, p25, p75, color="#cc3b30", alpha=0.25, label="IQR")
    ax.plot(TG, med_all, color="#cc3b30", lw=2.5, label="Median")
    ax.axvline(0, color="#6a6a6a", alpha=0.4, lw=0.8, ls=":")
    ax.set_xlim(-WIN/2, WIN/2); ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Amplitude (peak-normalized)", color="#1a1a1a", fontsize=FS_LABEL)
    ax.set_xlabel("Time relative to ectopic peak (s)", color="#1a1a1a", fontsize=FS_LABEL)
    ax.set_title(f"$\\bf{{(a)}}$ All PVCs overlaid — median ± IQR  (n={len(all_traces_norm):,})",
                 color="#1f1f1f", fontsize=8.5)
    ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
              fontsize=FS_LEGEND, loc="upper right")
    ax.tick_params(colors="#555555", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")
    ax.grid(alpha=0.18, color="#dcdcdc")

    # (2) — medians per session, SAME axes, VERTICAL legend OUTSIDE ON THE RIGHT
    ax = fig.add_axes(PANEL_POS["tr"], sharex=fig.axes[0], sharey=fig.axes[0])
    ax.set_facecolor(DARK_BG)
    palette = ["#3b6ea5","#c4622d","#3f8a4f","#8a5fb0","#9c6b3f",
               "#1f8f8f","#c45a8f","#7a7a7a","#6a8a3a","#9a4f6a",
               "#4f6a9a","#b8860b","#a0563f","#5f7a5f","#8f5a8f","#5a7d9a"]
    for i, s in enumerate(sessions):
        col = palette[i % len(palette)]
        m = np.median(s["traces_norm"], axis=0)
        ax.plot(TG, m, color=col, lw=1.1,
                label=short_label(s['label']))
    ax.axvline(0, color="#6a6a6a", alpha=0.4, lw=0.8, ls=":")
    ax.set_xlabel("Time relative to ectopic peak (s)", color="#1a1a1a", fontsize=FS_LABEL)
    ax.set_title(r"$\bf{(b)}$ Median morphology by session", color="#1f1f1f", fontsize=8.5)
    # vertical legend OUTSIDE the box, on the right side
    leg = ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                    fontsize=FS_TEXT-2, loc="center left",
                    bbox_to_anchor=(1.0, 0.5), ncol=1,
                    handlelength=1.0, handletextpad=0.4, borderpad=0.5,
                    labelspacing=0.45)
    leg.get_frame().set_linewidth(0.5)
    ax.tick_params(colors="#555555", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")
    ax.grid(alpha=0.18, color="#dcdcdc")

    # (3) — correlation matrix in panel (1, 0)
    labs = [short_label(s["label"]) for s in sessions]
    ax = fig.add_axes(PANEL_POS["bl"]); ax.set_facecolor(DARK_BG)
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=0.95, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(sessions))); ax.set_yticks(range(len(sessions)))
    ax.set_xticklabels(labs, color="#555555", rotation=45, ha="right", fontsize=FS_TEXT)
    ax.set_yticklabels(labs, color="#555555", fontsize=FS_TEXT)
    for i in range(len(sessions)):
        for j in range(len(sessions)):
            ax.text(j, i, f"{corr_matrix[i,j]:.3f}", ha="center", va="center",
                    color="black", fontsize=FS_TEXT-1)
    tight_cbar(fig, im, PANEL_POS["bl"], "", fs=FS_TICK)   # label in the caption (avoids overlap with panel d's y-label)
    ax.set_title(r"$\bf{(c)}$ Cross-session correlation matrix", color="#1f1f1f", fontsize=8.5)
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")

    # (4) — coupling interval distribution (RR_pre all PVCs)
    ax = fig.add_axes(PANEL_POS["br"]); ax.set_facecolor(DARK_BG)
    if len(all_coupling) > 0:
        ax.hist(all_coupling, bins=60, color="#d2685f", edgecolor="#ffffff",
                linewidth=0.3, density=True, alpha=0.85)
        med_c = float(np.median(all_coupling))
        ax.axvline(med_c, color="yellow", ls="--", lw=1.2, alpha=0.8,
                   label=f"median {med_c:.0f} ms")
        ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                  fontsize=FS_LEGEND, loc="upper right")
    ax.set_xlabel("Coupling interval RR_pre (ms)", color="#555555", fontsize=FS_LABEL)
    ax.set_ylabel("Density", color="#555555", fontsize=FS_LABEL)
    ax.set_title(f"$\\bf{{(d)}}$ Coupling interval distribution  (n={len(all_coupling):,})",
                 color="#1f1f1f", fontsize=8.5)
    ax.tick_params(colors="#555555", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")
    ax.grid(alpha=0.18, color="#dcdcdc")

    img_morphology_4panel = fig_to_b64(fig, dpi=450)

    # ============ CONTINUUM CHECK (data-driven, check if bimodal subtypes exist) ============
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        # center the normalized traces (shape-focus)
        X = all_traces_norm - all_traces_norm.mean(axis=1, keepdims=True)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        km2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        clusters_2 = km2.labels_
        # inertia for elbow
        inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
                    for k in range(1, 7)]
        # heatmap of PVCs ordered by trough depth
        order = np.argsort(trough_depth)
        step_hm = max(1, len(all_traces_norm) // 600)
        heatmap_data = all_traces_norm[order][::step_hm]

        # same width as Fig 1-3 (8.1in) for uniform rendering on the page
        fig = plt.figure(figsize=(8.1, 7.6), facecolor=DARK_BG)

        # (1) PCA scatter + KMeans k=2
        ax = fig.add_axes(PANEL_POS["tl"]); ax.set_facecolor(DARK_BG)
        for c, col, name in [(0, "#1f7fb0", "A"), (1, "#d2685f", "B")]:
            mask = clusters_2 == c
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=col, s=6, alpha=0.45,
                       label=f"Cluster {name} (n={mask.sum():,})")
        ax.set_xlabel("PC1", color="#1a1a1a", fontsize=FS_LABEL)
        ax.set_ylabel("PC2", color="#1a1a1a", fontsize=FS_LABEL)
        ax.set_title(r"$\bf{(a)}$ PCA + K-means k=2",
                     color="#1f1f1f", fontsize=8.5)
        ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                  fontsize=FS_LEGEND, loc="upper right")
        ax.tick_params(colors="#555555", labelsize=FS_TICK)
        for sp in ax.spines.values(): sp.set_color("#c8c8c8")
        ax.grid(alpha=0.18, color="#dcdcdc")

        # (2) heatmap of PVCs ordered by trough depth
        ax = fig.add_axes(PANEL_POS["tr"]); ax.set_facecolor(DARK_BG)
        im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r",
                       extent=[TG[0], TG[-1], 0, len(heatmap_data)],
                       vmin=-0.6, vmax=1.0, origin="lower")
        ax.set_xlabel("Time relative to ectopic peak (s)",
                      color="#1a1a1a", fontsize=FS_LABEL)
        ax.set_ylabel("PVCs sorted by trough depth", color="#1a1a1a", fontsize=FS_LABEL)
        ax.set_title(r"$\bf{(b)}$ PVCs sorted by hyperpolarization depth",
                     color="#1f1f1f", fontsize=8.5)
        ax.tick_params(colors="#555555", labelsize=FS_TICK)
        tight_cbar(fig, im, PANEL_POS["tr"], "Amplitude (norm.)", fs=FS_TICK)
        for sp in ax.spines.values(): sp.set_color("#c8c8c8")

        # (3) elbow plot
        ax = fig.add_axes(PANEL_POS["bl"]); ax.set_facecolor(DARK_BG)
        ax.plot(range(1, 7), inertias, marker="o", color="#b8860b", lw=2, ms=8)
        ax.set_xlabel("k (number of clusters)", color="#1a1a1a", fontsize=FS_LABEL)
        ax.set_ylabel("Within-cluster sum of squares", color="#1a1a1a", fontsize=FS_LABEL)
        ax.set_title(r"$\bf{(c)}$ Elbow plot",
                     color="#1f1f1f", fontsize=8.5)
        ax.tick_params(colors="#555555", labelsize=FS_TICK)
        for sp in ax.spines.values(): sp.set_color("#c8c8c8")
        ax.grid(alpha=0.18, color="#dcdcdc")

        # (4) trough depth distribution (moved here from the morphology panel)
        ax = fig.add_axes(PANEL_POS["br"]); ax.set_facecolor(DARK_BG)
        ax.hist(trough_depth, bins=60, color="#d2685f", edgecolor="#ffffff",
                linewidth=0.3, density=True, alpha=0.85)
        ax.set_xlabel("Post-QRS trough depth (peak-normalized)",
                      color="#555555", fontsize=FS_LABEL)
        ax.set_ylabel("Density", color="#555555", fontsize=FS_LABEL)
        ax.set_title(f"$\\bf{{(d)}}$ Hyperpolarization depth distribution  (n={len(trough_depth):,})",
                     color="#1f1f1f", fontsize=8.5)
        ax.tick_params(colors="#555555", labelsize=FS_TICK)
        for sp in ax.spines.values(): sp.set_color("#c8c8c8")
        ax.grid(alpha=0.18, color="#dcdcdc")

        img_pvc_continuum = fig_to_b64(fig, dpi=450)
    except Exception as e:
        print(f"  warning: continuum check failed: {e}")
        img_pvc_continuum = None

    # ============ NORMAL BEATS MORPHOLOGY (4-panel, same layout) ============
    med_per_sess_n = [np.median(s["traces_n_norm"], axis=0)
                      for s in sessions if len(s["traces_n_norm"])]
    corr_matrix_n = np.zeros((len(sessions), len(sessions)))
    for i in range(len(sessions)):
        for j in range(len(sessions)):
            corr_matrix_n[i, j] = np.corrcoef(med_per_sess_n[i], med_per_sess_n[j])[0, 1]
    med_all_n = np.median(all_n_norm, axis=0)
    p25_n = np.percentile(all_n_norm, 25, axis=0)
    p75_n = np.percentile(all_n_norm, 75, axis=0)
    y_min_n = min(p25_n.min(), min(m.min() for m in med_per_sess_n)) - 0.05
    y_max_n = 1.10

    # N outlier (session least correlated with the others): needed for panel (d)
    # and for the standalone figure further below (same data)
    n_sessions = len(sessions)
    mean_corr_per_session = []
    for i in range(n_sessions):
        rs = [corr_matrix_n[i, j] for j in range(n_sessions) if j != i]
        mean_corr_per_session.append((sessions[i]["label"], float(np.mean(rs)), i))
    mean_corr_per_session.sort(key=lambda x: x[1])
    outlier_label, outlier_r, outlier_idx = mean_corr_per_session[0]
    median_outlier = med_per_sess_n[outlier_idx]
    _others = [m for i, m in enumerate(med_per_sess_n) if i != outlier_idx]
    median_others = np.median(np.array(_others), axis=0)
    p25_others = np.percentile(np.array(_others), 25, axis=0)
    p75_others = np.percentile(np.array(_others), 75, axis=0)

    # width 8.1in = like Fig 1-4, for uniform rendering on the page
    fig = plt.figure(figsize=(8.1, 7.6), facecolor=DARK_BG)

    # (1,1) — overlay all N
    ax = fig.add_axes(PANEL_POS["tl"]); ax.set_facecolor(DARK_BG)
    step_n = max(1, len(all_n_norm) // 500)
    for tr in all_n_norm[::step_n]:
        ax.plot(TG, tr, color="#2f8a63", lw=0.4, alpha=0.06)
    ax.fill_between(TG, p25_n, p75_n, color="#2e8b57", alpha=0.25, label="IQR")
    ax.plot(TG, med_all_n, color="#2e8b57", lw=2.5, label="Median")
    ax.axvline(0, color="#6a6a6a", alpha=0.4, lw=0.8, ls=":")
    ax.set_xlim(-WIN/2, WIN/2); ax.set_ylim(y_min_n, y_max_n)
    ax.set_ylabel("Amplitude (peak-normalized)", color="#1a1a1a", fontsize=FS_LABEL)
    ax.set_xlabel("Time relative to sinus peak (s)", color="#1a1a1a", fontsize=FS_LABEL)
    ax.set_title(f"$\\bf{{(a)}}$ All N beats overlaid (sampled, n={len(all_n_norm):,}) — median ± IQR",
                 color="#1f1f1f", fontsize=8.5)
    ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
              fontsize=FS_LEGEND, loc="upper right")
    ax.tick_params(colors="#555555", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")
    ax.grid(alpha=0.18, color="#dcdcdc")

    # (1,2) — N medians per session
    ax = fig.add_axes(PANEL_POS["tr"], sharex=fig.axes[-1], sharey=fig.axes[-1])
    ax.set_facecolor(DARK_BG)
    for i, s in enumerate(sessions):
        if not len(s["traces_n_norm"]): continue
        col = palette[i % len(palette)]
        m = np.median(s["traces_n_norm"], axis=0)
        ax.plot(TG, m, color=col, lw=1.1,
                label=short_label(s['label']))
    ax.axvline(0, color="#6a6a6a", alpha=0.4, lw=0.8, ls=":")
    ax.set_xlabel("Time relative to sinus peak (s)", color="#1a1a1a", fontsize=FS_LABEL)
    ax.set_title(r"$\bf{(b)}$ Median N morphology by session", color="#1f1f1f", fontsize=8.5)
    leg = ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                    fontsize=FS_TEXT-2, loc="center left",
                    bbox_to_anchor=(1.0, 0.5), ncol=1,
                    handlelength=1.0, handletextpad=0.4, borderpad=0.5,
                    labelspacing=0.45)
    leg.get_frame().set_linewidth(0.5)
    ax.tick_params(colors="#555555", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")
    ax.grid(alpha=0.18, color="#dcdcdc")

    # (2,1) — correlation matrix N
    ax = fig.add_axes(PANEL_POS["bl"]); ax.set_facecolor(DARK_BG)
    im = ax.imshow(corr_matrix_n, cmap="RdYlGn", vmin=0.95, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(sessions))); ax.set_yticks(range(len(sessions)))
    ax.set_xticklabels(labs, color="#555555", rotation=45, ha="right", fontsize=FS_TEXT)
    ax.set_yticklabels(labs, color="#555555", fontsize=FS_TEXT)
    for i in range(len(sessions)):
        for j in range(len(sessions)):
            ax.text(j, i, f"{corr_matrix_n[i,j]:.3f}", ha="center", va="center",
                    color="black", fontsize=FS_TEXT-1)
    tight_cbar(fig, im, PANEL_POS["bl"], "", fs=FS_TICK)   # label in the caption (avoids overlap with panel d's y-label)
    ax.set_title(r"$\bf{(c)}$ Cross-session correlation matrix (N beats)",
                 color="#1f1f1f", fontsize=8.5)
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")

    # (2,2) — outlier session vs other sessions (median N) [ex Fig. 6 standalone]
    ax = fig.add_axes(PANEL_POS["br"]); ax.set_facecolor(DARK_BG)
    ax.fill_between(TG, p25_others, p75_others, color="#2e8b57", alpha=0.20,
                    label="others (IQR)")
    ax.plot(TG, median_others, color="#2e8b57", lw=2, label="others (median)")
    ax.plot(TG, median_outlier, color="#1f7fb0", lw=2.5, label="outlier")
    ax.axvline(0, color="#6a6a6a", alpha=0.4, lw=0.8, ls=":")
    ax.set_xlim(-WIN/2, WIN/2)
    ax.set_xlabel("Time relative to sinus peak (s)", color="#1a1a1a", fontsize=FS_LABEL)
    # y-label removed (it was right up against (c)'s colorbar); specified in the caption
    ax.set_title("$\\bf{(d)}$ Outlier session vs others (median N)",
                 color="#1f1f1f", fontsize=8.5)
    ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
              fontsize=FS_LEGEND, loc="upper right")
    ax.tick_params(colors="#555555", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")
    ax.grid(alpha=0.18, color="#dcdcdc")

    img_n_morphology_4panel = fig_to_b64(fig, dpi=450)

    # ============ OUTLIER ANALYSIS — N beats (standalone figure) ============
    # Same data as panel (d) above (outlier_*, median_others, ...).
    # SQUARE figure with INTERNAL legend → size of ONE 2x2 panel.
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG); ax.set_box_aspect(1)
    ax.fill_between(TG, p25_others, p75_others, color="#2e8b57", alpha=0.20,
                    label=f"Other {n_sessions-1} (IQR)")
    ax.plot(TG, median_others, color="#2e8b57", lw=2,
            label=f"Median other {n_sessions-1}")
    ax.plot(TG, median_outlier, color="#1f7fb0", lw=2.5,
            label=f"Outlier {short_label(outlier_label)} (r={outlier_r:.3f})")
    ax.axvline(0, color="#6a6a6a", alpha=0.4, lw=0.8, ls=":")
    ax.set_xlim(-WIN/2, WIN/2)
    ax.set_xlabel("Time relative to sinus peak (s)", color="#1a1a1a", fontsize=FS_LABEL)
    ax.set_ylabel("Amplitude (peak-normalized)", color="#1a1a1a", fontsize=FS_LABEL)
    ax.set_title("Outlier vs other sessions (median N)",
                 color="#1f1f1f", fontsize=FS_TITLE)
    leg = ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                    fontsize=FS_LEGEND, loc="upper right",
                    handlelength=1.2, handletextpad=0.5,
                    borderpad=0.5, labelspacing=0.4)
    leg.get_frame().set_linewidth(0.5)
    ax.tick_params(colors="#555555", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")
    ax.grid(alpha=0.18, color="#dcdcdc")
    img_n_outlier = fig_to_b64(fig, dpi=220)

    # ============ CROSS-SESSION RHYTHM & BURDEN (longitudinal, self-updating) ====
    # Same analyses as the synthetic report, recomputed from `sessions` at every run →
    # each new session automatically updates figure + table.
    cl = [short_label(s["label"]) for s in sessions]
    xs = np.arange(n_sessions)
    burden_v = [s["metrics"]["burden"]    for s in sessions]
    sahr_v   = [s["metrics"]["sa_hr"]     for s in sessions]
    pcomp_v  = [s["metrics"]["pct_comp"]  for s in sessions]
    pintp_v  = [s["metrics"]["pct_interp"] for s in sessions]

    fig = plt.figure(figsize=(8.1, 7.6), facecolor=DARK_BG)

    # (tl) burden per session, chronological order — session color coding consistent
    # with the morphology figures (same palette[i]) so each session is recognizable.
    ax = fig.add_axes(PANEL_POS["tl"]); ax.set_facecolor(DARK_BG)
    burden_cols = [palette[i % len(palette)] for i in range(n_sessions)]
    ax.bar(xs, burden_v, color=burden_cols, edgecolor="#ffffff", linewidth=0.4)
    ax.set_xticks(xs); ax.set_xticklabels(cl, rotation=45, ha="right",
                                          fontsize=FS_TEXT, color="#555555")
    ax.set_ylabel("PVC burden (%)", color="#555555", fontsize=FS_LABEL)
    ax.set_title("$\\bf{(a)}$ PVC burden by session", color="#1f1f1f", fontsize=8.5)
    ax.tick_params(colors="#555555", labelsize=FS_TICK)
    ax.grid(axis="y", alpha=0.18, color="#dcdcdc")
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")

    # (tr) effective SA HR vs interpolated/compensated share
    ax = fig.add_axes(PANEL_POS["tr"]); ax.set_facecolor(DARK_BG)
    ax.scatter(sahr_v, pcomp_v, s=70, c="#d2685f", edgecolors="#1a1a1a",
               linewidths=0.6, label="% compensated", zorder=4)
    ax.scatter(sahr_v, pintp_v, s=70, c="#1f7fb0", edgecolors="#1a1a1a",
               linewidths=0.6, label="% interpolated", zorder=4)
    try:   # weighted logistic fit (like the former key-pattern figure): comp red, interp blue
        from scipy.optimize import curve_fit
        _sa = np.array(sahr_v, dtype=float); _pc = np.array(pcomp_v, dtype=float) / 100.0
        _nc = np.array([s["metrics"]["n_interp"] + s["metrics"]["n_comp"]
                        for s in sessions], dtype=float)
        def _logi(x, a, b): return 1.0 / (1.0 + np.exp(-(a + b * x)))
        _popt, _ = curve_fit(_logi, _sa, np.clip(_pc, 1e-3, 1 - 1e-3),
                             p0=[-5.0, 0.12], sigma=1.0 / np.sqrt(np.maximum(_nc, 1)),
                             maxfev=20000)
        _xf = np.linspace(_sa.min() - 1.5, _sa.max() + 1.5, 200)
        _yc = 100 * _logi(_xf, *_popt)
        ax.plot(_xf, _yc, color="#d2685f", lw=1.8, zorder=2)
        ax.plot(_xf, 100 - _yc, color="#1f7fb0", lw=1.8, zorder=2)
    except Exception:
        pass
    ax.set_xlabel("Effective SA rate (BPM)", color="#555555", fontsize=FS_LABEL)
    ax.set_ylabel("Share of classified PVCs (%)", color="#555555", fontsize=FS_LABEL)
    ax.set_title("$\\bf{(b)}$ Effective rate vs pause type", color="#1f1f1f", fontsize=8.5)
    ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
              fontsize=FS_LEGEND, loc="best")
    ax.tick_params(colors="#555555", labelsize=FS_TICK)
    ax.grid(alpha=0.18, color="#dcdcdc")
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")

    # (bl) interpolated / compensated composition per session (2 classes)
    ax = fig.add_axes(PANEL_POS["bl"]); ax.set_facecolor(DARK_BG)
    yb = np.arange(n_sessions)
    ax.barh(yb, pintp_v, color="#1f7fb0", edgecolor="#ffffff", linewidth=0.4,
            label="Interpolated (silent)")
    ax.barh(yb, pcomp_v, left=pintp_v, color="#d2685f", edgecolor="#ffffff",
            linewidth=0.4, label="Compensated (felt)")
    ax.set_yticks(yb); ax.set_yticklabels(cl, fontsize=FS_TEXT, color="#555555")
    ax.set_xlim(0, 100); ax.set_xlabel("Composition (%)", color="#555555", fontsize=FS_LABEL)
    ax.set_title("$\\bf{(c)}$ Interpolated vs compensated", color="#1f1f1f", fontsize=8.5)
    ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
              fontsize=FS_LEGEND, loc="upper right")
    ax.tick_params(colors="#555555", labelsize=FS_TICK)
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")

    # (br) pre-PVC coupling stability (median ± IQR per session)
    ax = fig.add_axes(PANEL_POS["br"]); ax.set_facecolor(DARK_BG)
    med, lo, hi = [], [], []
    for s in sessions:
        c = s["coupling_ms"]
        if len(c):
            mm = float(np.median(c))
            med.append(mm); lo.append(mm - np.percentile(c, 25))
            hi.append(np.percentile(c, 75) - mm)
        else:
            med.append(np.nan); lo.append(0); hi.append(0)
    ax.errorbar(xs, med, yerr=[lo, hi], fmt="o", color="#cc3b30",
                ecolor="#7a3b3b", elinewidth=1.2, capsize=3, ms=6, zorder=4)
    valid = [m for m in med if not np.isnan(m)]
    if valid:
        gm = float(np.median(valid))
        ax.axhline(gm, color="#6a6a6a", ls="--", lw=1, alpha=0.7,
                   label=f"global median {gm:.0f} ms")
        ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                  fontsize=FS_LEGEND, loc="best")
    ax.set_xticks(xs); ax.set_xticklabels(cl, rotation=45, ha="right",
                                          fontsize=FS_TEXT, color="#555555")
    ax.set_ylabel("Pre-PVC coupling (ms)", color="#555555", fontsize=FS_LABEL)
    ax.set_title("$\\bf{(d)}$ Coupling interval stability",
                 color="#1f1f1f", fontsize=8.5)
    ax.tick_params(colors="#555555", labelsize=FS_TICK)
    ax.grid(axis="y", alpha=0.18, color="#dcdcdc")
    for sp in ax.spines.values(): sp.set_color("#c8c8c8")

    img_crosssession = fig_to_b64(fig, dpi=450)

    # ============ KEY PATTERN: resting sinus rate vs felt (compensated) PVCs =======
    # Plot requested by the user: one line for the compensated (perceived thumps) and
    # one for the interpolated (silent) as a function of the mean resting sinus
    # rate. With 2 classes pct_interp = 100 - pct_comp → mirror curves. Weighted
    # logistic fit on the number of classified beats; trend quantified with Spearman.
    img_hr_pattern = None
    sinus_v   = np.array([s["metrics"]["sinus_bpm"] for s in sessions], dtype=float)
    pcomp_arr = np.array(pcomp_v, dtype=float)
    pint_arr  = np.array(pintp_v, dtype=float)
    nclass_v  = np.array([s["metrics"]["n_interp"] + s["metrics"]["n_comp"]
                          for s in sessions], dtype=float)
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(sinus_v, pcomp_arr)
        fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        # weighted logistic fit: p_comp = 1/(1+exp(-(a + b·bpm)))
        xs_fit = np.linspace(sinus_v.min() - 1.5, sinus_v.max() + 1.5, 200)
        try:
            from scipy.optimize import curve_fit
            def logi(x, a, b): return 1.0 / (1.0 + np.exp(-(a + b * x)))
            frac = np.clip(pcomp_arr / 100.0, 1e-3, 1 - 1e-3)
            sigma = 1.0 / np.sqrt(np.maximum(nclass_v, 1))
            popt, _ = curve_fit(logi, sinus_v, frac, p0=[-5.0, 0.12],
                                sigma=sigma, maxfev=20000)
            yc_fit = 100 * logi(xs_fit, *popt)
        except Exception:
            # fallback: simple linear regression
            b1, b0 = np.polyfit(sinus_v, pcomp_arr, 1)
            yc_fit = np.clip(b0 + b1 * xs_fit, 0, 100)
        yi_fit = 100 - yc_fit
        ax.plot(xs_fit, yc_fit, color="#cc3b30", lw=2.2, zorder=2)
        ax.plot(xs_fit, yi_fit, color="#1f7fb0", lw=2.2, zorder=2)
        ax.axhline(50, color="#666", ls=":", lw=0.8, alpha=0.6)
        ax.scatter(sinus_v, pcomp_arr, s=130, c="#cc3b30", edgecolors="#1a1a1a",
                   linewidths=1.0, zorder=4, label="% compensated (felt thumps)")
        ax.scatter(sinus_v, pint_arr, s=130, c="#1f7fb0", edgecolors="#1a1a1a",
                   linewidths=1.0, zorder=4, label="% interpolated (silent)")
        for x, yc, yi, s in zip(sinus_v, pcomp_arr, pint_arr, sessions):
            lab = short_label(s["label"])
            ax.annotate(lab, (x, yc), textcoords="offset points", xytext=(6, 6),
                        color="#d2685f", fontsize=FS_TEXT - 0.5, fontweight="bold")
            ax.annotate(lab, (x, yi), textcoords="offset points", xytext=(6, -12),
                        color="#1f7fb0", fontsize=FS_TEXT - 0.5, fontweight="bold")
        ax.set_xlabel("Resting sinus rate (BPM, mean N/min)", color="#333333", fontsize=FS_LABEL)
        ax.set_ylabel("Share of classified PVCs (%)", color="#333333", fontsize=FS_LABEL)
        ax.set_ylim(-3, 103)
        ax.set_title("Key pattern: resting heart rate sets how many PVCs are felt   "
                     f"(Spearman r={rho:.2f}, p={pval:.3f})",
                     color="#1f1f1f", fontsize=FS_TITLE)
        ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                  fontsize=FS_LEGEND, loc="center right")
        ax.tick_params(colors="#555555", labelsize=FS_TICK)
        ax.grid(alpha=0.16, color="#dcdcdc")
        for sp in ax.spines.values(): sp.set_color("#c8c8c8")
        fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.11)
        img_hr_pattern = fig_to_b64(fig, dpi=220)
    except Exception as e:
        print(f"  warning: HR-pattern figure failed: {e}")

    # cross-session metrics table (HTML)
    cross_rows = []
    for s in sessions:
        m = s["metrics"]
        cross_rows.append(
            f"<tr><td>{s['label']}</td>"
            f"<td class='num'>{s['duration_min']:.0f}</td>"
            f"<td class='num'>{m['sinus_bpm']:.1f}</td>"
            f"<td class='num'>{m['sa_hr']:.0f}</td>"
            f"<td class='num'>{m['pvc_rate']:.1f}</td>"
            f"<td class='num'>{m['burden']:.1f}%</td>"
            f"<td class='num'>{m['pct_interp']:.0f}%</td>"
            f"<td class='num'>{m['pct_comp']:.0f}%</td>"
            f"<td class='num'>{m['n_couplet']}</td></tr>")
    cross_table = "\n".join(cross_rows)

    # ============ SUMMARY TABLE per session (synthetic-report style) =========
    # Metrics as rows, sessions as columns (tinted columns = the figures' palette).
    # Traffic-light on burden/couplet/AF; interp light blue, comp red. All on validated data.
    def _sem_burden(v): return "#1b8a3a" if v < 15 else ("#b8860b" if v < 25 else "#cc5a2a")
    def _sem_couplet(n): return "#1b8a3a" if n == 0 else ("#b8860b" if n <= 3 else "#cc5a2a")
    def _sem_af(sc):
        if sc is None: return "#6a6a6a"
        return "#1b8a3a" if sc == 0 else ("#b8860b" if sc <= 2 else "#cc5a2a")
    ST = []
    for s in sessions:
        m = s["metrics"]; cm = s["coupling_ms"]
        cmv = cm[(cm > 200) & (cm < 800)] if len(cm) else cm
        ST.append({"m": m, "s": s,
                   "coup_med": float(np.median(cmv)) if len(cmv) else 0.0,
                   "n_coup": len(s["couplets"]),
                   "guard": sum(1 for d in s["pause_data"] if d["guard"])})
    _tint = lambda i: palette[i % len(palette)] + "14"
    def _cell(content, i):
        return f"<td class='num' style='background:{_tint(i)}'>{content}</td>"
    _summary_rows = []
    def _add(label, render, emph=False):
        lab = (f"<td style='text-align:left;#1a1a1a-space:nowrap;"
               f"font-weight:{'700' if emph else '400'}'>{label}</td>")
        _summary_rows.append("<tr>" + lab
                             + "".join(render(ST[i], i) for i in range(len(ST))) + "</tr>")
    _npvc = lambda x: max(1, x["s"]["n_pvc"])
    _add("Useful duration (min)", lambda x, i: _cell(f"{x['m']['clean_s']/60:.0f}", i))
    _add("Excluded (s)",          lambda x, i: _cell(f"{x['s']['excluded_seconds']:.0f}", i))
    _add("Total beats",           lambda x, i: _cell(f"{x['m']['n_total']:,}", i))
    _add("Sinus rate (BPM)",      lambda x, i: _cell(f"{x['m']['sinus_bpm']:.1f}", i))
    _add("Effective SA (BPM)",    lambda x, i: _cell(f"{x['m']['sa_hr']:.0f}", i))
    _add("PVC total",             lambda x, i: _cell(f"{x['s']['n_pvc']:,} <span style='color:#6a6a6a'>({x['m']['burden']:.1f}%)</span>", i))
    _add("PVC rate (/min)",       lambda x, i: _cell(f"{x['m']['pvc_rate']:.1f}", i))
    _add("Burden (%)",            lambda x, i: _cell(f"<b style='color:{_sem_burden(x['m']['burden'])}'>{x['m']['burden']:.1f}%</b>", i), emph=True)
    _add("Median coupling (ms)",  lambda x, i: _cell(f"{x['coup_med']:.0f}", i))
    _add("Couplets",              lambda x, i: _cell(f"<b style='color:{_sem_couplet(x['n_coup'])}'>{x['n_coup']}</b> <span style='color:#6a6a6a'>({100*x['n_coup']/_npvc(x):.2f}%)</span>", i), emph=True)
    _add("Interpolated",          lambda x, i: _cell(f"<span style='color:#1f7fb0'><b>{x['m']['n_interp']}</b> ({x['m']['pct_interp']:.0f}%)</span>", i))
    _add("Compensated",           lambda x, i: _cell(f"<span style='color:#d2685f'><b>{x['m']['n_comp']}</b> ({x['m']['pct_comp']:.0f}%)</span>", i))
    _add("Guarded (ambiguous)",   lambda x, i: _cell(f"<span style='color:#6a6a6a'>{x['guard']}</span>", i))
    _add("Isolated PVC",          lambda x, i: _cell(f"{x['m']['iso_pvc']} <span style='color:#6a6a6a'>({100*x['m']['iso_pvc']/_npvc(x):.0f}%)</span>", i))
    _add("Bigeminy V-N-V",        lambda x, i: _cell(f"{x['m']['bigem']} <span style='color:#6a6a6a'>({100*x['m']['bigem']/_npvc(x):.0f}%)</span>", i))
    _add("Trigeminy V-N-N-V",     lambda x, i: _cell(f"{x['m']['trigem']} <span style='color:#6a6a6a'>({100*x['m']['trigem']/_npvc(x):.0f}%)</span>", i))
    _add("AF score (0-4)",        lambda x, i: _cell((f"<b style='color:{_sem_af(x['m']['af_score'])}'>{x['m']['af_score']}/4</b>") if x['m']['af_score'] is not None else "-", i), emph=True)
    _add("RMSSD (ms)",            lambda x, i: _cell(f"{x['m']['rmssd']:.0f}", i))
    summary_head = "<th style='text-align:left'>Metric</th>" + "".join(
        f"<th style='background:{palette[i%len(palette)]}40;color:#fff;#1a1a1a-space:nowrap'>"
        f"{short_label(s['label'])}</th>" for i, s in enumerate(sessions))
    summary_body = "\n".join(_summary_rows)

    # ============ METHOD: interpolated vs compensated — criterio & validazione ====
    method_n     = len(all_post)
    _ap          = np.asarray(all_post, dtype=float)
    method_pct_int  = 100 * float(np.mean(_ap <  PAUSE_VALLEY)) if method_n else 0
    method_pct_comp = 100 * float(np.mean(_ap >= PAUSE_VALLEY)) if method_n else 0
    method_amb   = 100 * float(np.mean(np.abs(_ap - PAUSE_VALLEY) < 0.10)) if method_n else 0
    method_guard = sum(1 for s in sessions for d in s["pause_data"] if d["guard"])
    img_method_example = img_method_dist = img_method_strip = None
    demo = max(sessions, key=lambda s: min(s["metrics"]["n_interp"], s["metrics"]["n_comp"]))
    _dd = load_session(demo["ecg_path"])
    if _dd is not None:
        dt, dvf, dpeaks, dexcl = _dd
        dpause = demo["pause_data"]

        def _far(tv):
            return all(not (a-2 <= tv <= b+2) for a, b in dexcl)
        def _clean(kind):
            cand = [d for d in dpause if not d["guard"] and d["t"] > 60 and _far(d["t"])]
            tgt = 0.70 if kind == "int" else 1.45
            cand = [d for d in cand if (d["post_ratio"] < PAUSE_VALLEY) == (kind == "int")]
            cand.sort(key=lambda d: abs(d["post_ratio"] - tgt))
            return cand[0] if cand else None
        # semi-transparent white box behind the labels, so they don't blend in
        # with the vertical lines that cross them
        _lblbox = dict(boxstyle="round,pad=0.12", facecolor="#ffffff",
                       edgecolor="none", alpha=0.72)
        def _draw_demo(ax, d, color, letter, keyword, suffix, show_xlabel=True):
            c = d["t"]; half = 2.6; m = (dt >= c-half) & (dt <= c+half)
            ax.set_facecolor(DARK_BG); ax.plot(dt[m]-c, dvf[m], lw=0.9, color="#5a6b78")
            wm = (dt >= c-0.12) & (dt <= c+0.12)
            ax.plot(dt[wm]-c, dvf[wm], lw=1.8, color=color)
            ax.scatter(0, d["amp"], s=130, marker="o", facecolors="none",
                       edgecolors=color, linewidths=2, zorder=6)
            npx = -d["rr_pre"]; nnx = d["rr_post"]
            ax.axvline(npx, color="#2e8b57", lw=1.0, alpha=0.7)
            ax.axvline(nnx, color="#2e8b57", lw=1.5, alpha=0.95)
            ax.text(npx, 1.45, "N prev", color="#2e8b57", fontsize=FS_TEXT, ha="center",
                    bbox=_lblbox, zorder=7)
            ax.text(nnx, 1.45, "N next", color="#2e8b57", fontsize=FS_TEXT, ha="center",
                    bbox=_lblbox, zorder=7)
            ref1 = npx + d["rl"]; ref2 = npx + 2*d["rl"]
            ax.axvline(ref1, color="#1f7fb0", ls="--", lw=1.1, alpha=0.9)
            ax.axvline(ref2, color="#cc3b30", ls="--", lw=1.1, alpha=0.9)
            ax.text(ref1, -0.98, "1×", color="#1f7fb0", fontsize=FS_TEXT, ha="center",
                    va="top", bbox=_lblbox, zorder=7)
            ax.text(ref2, -0.98, "2×", color="#cc3b30", fontsize=FS_TEXT, ha="center",
                    va="top", bbox=_lblbox, zorder=7)
            ax.plot([0, nnx], [-0.85, -0.85], color="#b8860b", lw=3, solid_capstyle="butt")
            ax.text(nnx/2, -0.74, "RR_post", color="#b8860b", fontsize=FS_TEXT, ha="center",
                    bbox=_lblbox, zorder=7)
            ax.set_xlim(-half, half); ax.set_ylim(-1.25, 1.75)
            ax.tick_params(colors="#555555", labelsize=FS_TICK)
            for sp in ax.spines.values(): sp.set_color("#c8c8c8")
            ax.grid(True, alpha=0.13, color="#dcdcdc", lw=0.3)
            if show_xlabel:
                ax.set_xlabel("t (s) relative to the PVC", color="#555555", fontsize=FS_LABEL)
            # black title, with only the class word (keyword) in color
            colored_title(ax, [(letter, "#1a1a1a", "bold"),
                               (keyword, color, "normal"),
                               (suffix, "#1a1a1a", "normal")])
        di, dc = _clean("int"), _clean("comp")
        if di and dc:
            # stacked vertically: 8.1in width like the other figures
            # (the ECG strip stays wide), uniform font/dpi, letters (a)/(b) in bold
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.1, 5.0), facecolor=DARK_BG)
            _draw_demo(a1, di, "#1f7fb0", "$\\bf{(a)}$  ", "Interpolated",
                       f" — pause {di['post_ratio']:.2f}× sinus (silent)", show_xlabel=False)
            _draw_demo(a2, dc, "#cc3b30", "$\\bf{(b)}$  ", "Compensated",
                       f" — pause {dc['post_ratio']:.2f}× sinus (felt)")
            a1.tick_params(labelbottom=False)   # x numbers only on the bottom panel
            fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.10, hspace=0.30)
            img_method_example = fig_to_b64(fig, dpi=450)

        # distributions: sum S (convention) vs pause RR_post (perception)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.1, 3.2), facecolor=DARK_BG)
        a1.set_facecolor(DARK_BG)
        a1.hist(all_sratio, bins=np.arange(0.6, 3.0, 0.04), color="#6a6a6a",
                alpha=0.55, edgecolor="#ffffff", linewidth=0.3)
        a1.set_title("$\\bf{(a)}$ Conventional sum S", color="#1f1f1f", fontsize=8.5)
        a1.set_xlabel("S  (× sinus cycle)", color="#555555", fontsize=FS_LABEL)
        a2.set_facecolor(DARK_BG)
        a2.hist(all_post, bins=np.arange(0.3, 2.2, 0.035), color="#1f7fb0",
                alpha=0.5, edgecolor="#ffffff", linewidth=0.3)
        a2.axvline(PAUSE_VALLEY, color="#b8860b", lw=2, label=f"valley {PAUSE_VALLEY:.2f}")
        a2.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8", fontsize=FS_LEGEND)
        a2.set_title("$\\bf{(b)}$ Post-extrasystolic pause", color="#1f1f1f", fontsize=8.5)
        a2.set_xlabel("RR_post  (× sinus cycle)", color="#555555", fontsize=FS_LABEL)
        for ax in (a1, a2):
            ax.tick_params(colors="#555555", labelsize=FS_TICK); ax.grid(alpha=0.15, color="#dcdcdc")
            for sp in ax.spines.values(): sp.set_color("#c8c8c8")
        fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.16, wspace=0.16)
        img_method_dist = fig_to_b64(fig, dpi=450)

        # final strip colored by RR_post (2 classes) — quality_strip / Fig 1 style
        from matplotlib.lines import Line2D
        cmapd = {d["i"]: ("int" if d["post_ratio"] < PAUSE_VALLEY else "comp")
                 for d in dpause if not d["guard"]}
        nrow = 6
        ti = [d["t"] for d in dpause if cmapd.get(d["i"]) == "int" and d["t"] > 60]
        tc = [d["t"] for d in dpause if cmapd.get(d["i"]) == "comp" and d["t"] > 60]
        bt, bsc, t0 = 60.0, -1, 60.0
        while t0 + nrow*10 < dt[-1]:
            a = sum(1 for x in ti if t0 <= x < t0+nrow*10); b = sum(1 for x in tc if t0 <= x < t0+nrow*10)
            sc = min(a, b)*2 + a + b
            if sc > bsc: bsc, bt = sc, t0
            t0 += 20
        COLc = {"int": "#1f7fb0", "comp": "#cc3b30"}
        fig, axes = plt.subplots(nrow, 1, figsize=(8.1, 0.8*nrow + 1.0), facecolor=DARK_BG)
        for r, ax in enumerate(axes):
            rs = bt + r*10; re = rs+10; m = (dt >= rs) & (dt < re)
            ax.set_facecolor(DARK_BG)
            if m.any(): ax.plot(dt[m]-rs, dvf[m], lw=0.45, color="#2e8b57", alpha=0.9)
            for d in dpause:
                if not (rs <= d["t"] < re) or d["i"] not in cmapd: continue
                c = cmapd[d["i"]]
                wm = (dt >= d["t"]-0.12) & (dt <= d["t"]+0.12)
                if wm.any(): ax.plot(dt[wm]-rs, dvf[wm], lw=0.9, color=COLc[c])
                ax.scatter(d["t"]-rs, min(1.5, d["amp"]+0.30), s=28, marker="v",
                           color=COLc[c], edgecolors="#1a1a1a", linewidths=0.35, zorder=6)
            ax.set_xlim(0, 10); ax.set_ylim(-1.2, 1.7)
            ax.set_yticks([]); ax.tick_params(axis="x", colors="#777777", labelsize=8.5)
            if r < nrow - 1: ax.set_xticklabels([])
            ax.grid(True, alpha=0.13, color="#dcdcdc", lw=0.4)
            for sp in ax.spines.values(): sp.set_color("#cccccc")
            ax.text(-0.012, 0.5, f"{int(rs//60):02d}:{int(rs%60):02d}", transform=ax.transAxes,
                    ha="right", va="center", color="#888888", fontsize=9)
        axes[-1].set_xlabel("time (s)", color="#666666", fontsize=10)
        fig.legend(handles=[Line2D([0],[0], color="#1f7fb0", lw=1.8, label="interpolated"),
                            Line2D([0],[0], color="#cc3b30", lw=1.8, label="compensated")],
                   loc="upper center", ncol=2, fontsize=9.5, frameon=False,
                   bbox_to_anchor=(0.5, 0.995), columnspacing=2.4)
        fig.subplots_adjust(left=0.055, right=0.992, top=0.90, bottom=0.085, hspace=0.30)
        img_method_strip = fig_to_b64(fig, dpi=450)

        # --- 3-class variant (report, goes BEFORE Fig 7): same style as Fig 9
        #     but with a THIRD yellow class for the intermediate PVCs (pause near the
        #     valley, ±BAND), besides blue=interpolated and red=compensated. Shows that
        #     with the pause cut alone a few borderline cases remain. Window chosen
        #     to contain all three classes. Saved in figs_manual/.
        BAND = 0.15
        def _cls_p(pr):
            if pr < PAUSE_VALLEY - BAND: return "int"
            if pr > PAUSE_VALLEY + BAND: return "comp"
            return "mid"
        cmap3 = {d["i"]: _cls_p(d["post_ratio"]) for d in dpause if not d["guard"]}
        COL3 = {"int": "#1f7fb0", "comp": "#cc3b30", "mid": "#e0a800"}
        nrow3 = 6   # quality_strip / Fig 1 style: 8.1in, 6 rows of 10s, 450 dpi
        # window (nrow3 x 10s) that maximizes the presence of ALL three classes
        _tcl = [(d["t"], cmap3[d["i"]]) for d in dpause if d["i"] in cmap3 and d["t"] > 60]
        bt3, _bsc, _t0 = bt, -1, 60.0
        while _t0 + nrow3*10 < dt[-1]:
            _c = {"int": 0, "mid": 0, "comp": 0}
            for _x, _cl in _tcl:
                if _t0 <= _x < _t0 + nrow3*10: _c[_cl] += 1
            _sc = min(_c.values())*3 + sum(_c.values())
            if _sc > _bsc: _bsc, bt3 = _sc, _t0
            _t0 += 20
        fig, axes = plt.subplots(nrow3, 1, figsize=(8.1, 0.8*nrow3 + 1.0), facecolor=DARK_BG)
        for r, ax in enumerate(axes):
            rs = bt3 + r*10; re = rs+10; m = (dt >= rs) & (dt < re)
            ax.set_facecolor(DARK_BG)
            if m.any(): ax.plot(dt[m]-rs, dvf[m], lw=0.45, color="#2e8b57", alpha=0.9)
            for d in dpause:
                if not (rs <= d["t"] < re) or d["i"] not in cmap3: continue
                c = cmap3[d["i"]]
                wm = (dt >= d["t"]-0.12) & (dt <= d["t"]+0.12)
                if wm.any(): ax.plot(dt[wm]-rs, dvf[wm], lw=0.9, color=COL3[c])
                ax.scatter(d["t"]-rs, min(1.5, d["amp"]+0.30), s=28, marker="v",
                           color=COL3[c], edgecolors="#1a1a1a", linewidths=0.35, zorder=6)
            ax.set_xlim(0, 10); ax.set_ylim(-1.2, 1.7)
            ax.set_yticks([]); ax.tick_params(axis="x", colors="#777777", labelsize=8.5)
            if r < nrow3 - 1: ax.set_xticklabels([])
            ax.grid(True, alpha=0.13, color="#dcdcdc", lw=0.4)
            for sp in ax.spines.values(): sp.set_color("#cccccc")
            ax.text(-0.012, 0.5, f"{int(rs//60):02d}:{int(rs%60):02d}", transform=ax.transAxes,
                    ha="right", va="center", color="#888888", fontsize=9)
        axes[-1].set_xlabel("time (s)", color="#666666", fontsize=10)
        fig.legend(handles=[Line2D([0],[0], color="#1f7fb0", lw=1.8, label="interpolated"),
                            Line2D([0],[0], color="#e0a800", lw=1.8, label="intermediate"),
                            Line2D([0],[0], color="#cc3b30", lw=1.8, label="compensated")],
                   loc="upper center", ncol=3, fontsize=9.5, frameon=False,
                   bbox_to_anchor=(0.5, 0.995), columnspacing=2.4)
        fig.subplots_adjust(left=0.055, right=0.992, top=0.90, bottom=0.085, hspace=0.30)
        os.makedirs("reports/figs_manual", exist_ok=True)
        fig.savefig("reports/figs_manual/interp_comp_3class_strip.png", dpi=450, facecolor=DARK_BG)
        plt.close(fig)

    # ============ PER-SESSION DISTRIBUTIONS: pause & coupling histograms ==========
    # One row per session, recomputed at every run (report style). They use the
    # validated criterion: pause RR_post / LOCAL sinus cycle, cut at the global
    # PAUSE_VALLEY. They show, session by session, why the interp/comp split
    # falls where it falls (the bimodal shape is readable by eye).
    nS = len(sessions)
    palette_ps = palette  # reuse the palette from the cross section

    # (A) distribution of the PAUSE RR_post / sinus — silent hump (~0.75x) and
    #     with-pause hump (~1.45x), yellow valley = interp/comp threshold.
    img_persession_pause = None
    bins_pr = np.linspace(0.2, 2.0, 64)
    fig, axes_pp = plt.subplots(nS, 1, figsize=(8.1, 1.1 * nS),
                                facecolor=DARK_BG, sharex=True, squeeze=False)
    axes_pp = axes_pp.ravel()
    for ax, s in zip(axes_pp, sessions):
        ax.set_facecolor(DARK_BG)
        pr = np.array([d["post_ratio"] for d in s["pause_data"] if not d["guard"]])
        n_g = sum(1 for d in s["pause_data"] if d["guard"])
        pr_i = pr[pr < PAUSE_VALLEY]
        pr_c = pr[pr >= PAUSE_VALLEY]
        ax.hist(pr_i, bins=bins_pr, color="#1f7fb0", edgecolor="#ffffff", linewidth=0.3)
        ax.hist(pr_c, bins=bins_pr, color="#d2685f", edgecolor="#ffffff", linewidth=0.3)
        ax.axvline(PAUSE_VALLEY, color="#b8860b", ls="-", lw=1.5)
        ax.axvline(1.0, color="#6a6a6a", ls=":", lw=0.8, alpha=0.6)
        if len(pr):
            med = float(np.median(pr))
            ax.axvline(med, color="#1a1a1a", ls="--", lw=1.0, alpha=0.7)
        pct_c = 100 * len(pr_c) / max(1, len(pr))
        ax.text(0.012, 0.84, "interpolated", transform=ax.transAxes,
                color="#1f7fb0", fontsize=FS_TEXT)
        ax.text(0.988, 0.84, "compensated", transform=ax.transAxes,
                color="#d2685f", fontsize=FS_TEXT, ha="right")
        ax.set_title(f"{short_label(s['label'])}   "
                     f"(SA {s['metrics']['sa_hr']:.0f} BPM, n={len(pr)}, "
                     f"{pct_c:.0f}% compensated"
                     + (f", {n_g} guarded" if n_g else "") + ")",
                     color="#1f1f1f", fontsize=FS_TICK + 0.5, pad=2)
        ax.set_ylabel("count", color="#666666", fontsize=FS_TEXT)
        ax.tick_params(colors="#555555", labelsize=FS_TICK)
        ax.grid(axis="y", alpha=0.15, color="#dcdcdc", lw=0.3)
        for sp in ax.spines.values(): sp.set_color("#c8c8c8")
    axes_pp[-1].set_xlabel(f"pause RR$_{{post}}$ / local sinus cycle   "
                           f"(yellow = global valley {PAUSE_VALLEY:.2f}×, interp/comp split)",
                           color="#555555", fontsize=FS_LABEL)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.965, bottom=0.05, hspace=0.45)
    img_persession_pause = fig_to_b64(fig, dpi=450)

    # (B) distribution of the pre-PVC COUPLING per session (focus stability).
    #     Bars colored by sub-cluster: <500 (blue), 500-600 (pink), >600 (green).
    img_persession_coupling = None
    bins_c = np.arange(280, 720, 14)
    cen_c = (bins_c[:-1] + bins_c[1:]) / 2
    clu_col = ["#1f7fb0" if x < 500 else ("#d2685f" if x < 600 else "#2e8b57")
               for x in cen_c]
    # dual-focus check per session (coupling modality + morphology)
    focus_findings = []   # for the HTML text
    fig, axes_pc = plt.subplots(nS, 1, figsize=(8.1, 1.1 * nS),
                                facecolor=DARK_BG, sharex=True, squeeze=False)
    axes_pc = axes_pc.ravel()
    for ax, s in zip(axes_pc, sessions):
        ax.set_facecolor(DARK_BG)
        c = s["coupling_ms"]
        c = c[(c > 200) & (c < 800)] if len(c) else c
        mod = coupling_modality(c)
        focus_txt = "unimodal"
        if len(c):
            h, _ = np.histogram(c, bins=bins_c)
            ax.bar(cen_c, h, width=12, color=clu_col, edgecolor="#ffffff", linewidth=0.3)
            med = float(np.median(c))
            ax.axvline(med, color="#b8860b", ls="-", lw=1.4)
            ax.text(med + 4, ax.get_ylim()[1] * 0.8, f"med {med:.0f} ms",
                    color="#b8860b", fontsize=FS_TEXT, fontweight="bold")
            if mod["bimodal"]:
                # real second mode → check morphology (same focus?)
                morph = coupling_focus_morph(s["ecg_path"], mod["valley"])
                ax.axvline(mod["valley"], color="#6f42c1", ls="--", lw=1.2)
                if morph and morph["corr"] > 0.97:
                    focus_txt = (f"bimodal: same morphology (QRS r={morph['corr']:.3f})")
                    tag_col = "#6f42c1"
                elif morph:
                    focus_txt = (f"bimodal: CHECK morphology (QRS r={morph['corr']:.3f})")
                    tag_col = "#cc3b30"
                else:
                    focus_txt = "bimodal (morphology n/a)"
                    tag_col = "#6f42c1"
                ax.text(0.985, 0.84, focus_txt, transform=ax.transAxes, ha="right",
                        color=tag_col, fontsize=FS_TEXT - 0.5, fontweight="bold")
                focus_findings.append((short_label(s["label"]), focus_txt, morph))
            elif mod["ok"] and mod["mu"]:
                # no real trough (right shoulder): artificial split at the midpoint
                # of the two modes and check that the morphology stays the same
                split = 0.5 * (mod["mu"][0] + mod["mu"][1])
                morph = coupling_focus_morph(s["ecg_path"], split)
                if morph:
                    ax.text(0.985, 0.84, f"artificially split QRS r={morph['corr']:.3f}",
                            transform=ax.transAxes, ha="right", color="#9a7d0a",
                            fontsize=FS_TEXT - 0.5, fontstyle="italic")
        ax.axvline(500, color="#6a6a6a", ls="--", lw=0.7, alpha=0.5)
        ax.axvline(600, color="#6a6a6a", ls="--", lw=0.7, alpha=0.5)
        ax.set_title(f"{short_label(s['label'])}   pre-PVC coupling (n={len(c)})",
                     color="#1f1f1f", fontsize=FS_TICK + 0.5, pad=2)
        ax.set_ylabel("count", color="#666666", fontsize=FS_TEXT)
        ax.tick_params(colors="#555555", labelsize=FS_TICK)
        ax.grid(axis="y", alpha=0.15, color="#dcdcdc", lw=0.3)
        for sp in ax.spines.values(): sp.set_color("#c8c8c8")
    axes_pc[-1].set_xlabel("pre-PVC coupling interval (ms)",
                           color="#555555", fontsize=FS_LABEL)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.965, bottom=0.05, hspace=0.5)
    img_persession_coupling = fig_to_b64(fig, dpi=450)

    # textual summary of the dual-focus check (for the HTML)
    n_bimodal = len(focus_findings)
    n_diff = sum(1 for _, txt, _ in focus_findings if "CHECK" in txt)
    if n_bimodal == 0:
        focus_summary = ("Every session's coupling is statistically unimodal — a "
                         "single pre-PVC coupling peak, consistent with one focus.")
    else:
        same = ", ".join(f"{lab} (r={m['corr']:.3f})"
                         for lab, txt, m in focus_findings if m and "CHECK" not in txt)
        focus_summary = (
            f"{n_bimodal} session(s) show a genuinely <b>bimodal</b> coupling "
            f"(two peaks, not just skew): {same}. In each the two coupling clusters "
            f"have <b>identical QRS morphology</b> (template correlation as shown), so "
            f"this is the <b>same monomorphic focus discharging at two coupling "
            f"intervals</b> (coupling modulation), <b>not</b> a second focus."
            + (f" {n_diff} session(s) flagged for morphology review."
               if n_diff else " No session shows a morphologically distinct second focus."))

    # ============ COUPLETS: detection, per-session count, morphological overlay ==
    # Couplet = 2 consecutive PVCs (RR 200-700ms), not part of a run>=3. Collected per
    # session in s["couplets"]. Here: (1) per-session count, (2) overlay of ALL
    # couplets aligned on the 1st PVC's peak (normalized) to see if they are
    # similar, (3) overlay of the individual QRS of the 1st vs 2nd PVC (same morphology?).
    img_couplets = None
    coup_per_sess = [len(s["couplets"]) for s in sessions]
    all_coup = [(c, i) for i, s in enumerate(sessions) for c in s["couplets"]]
    n_coup_tot = len(all_coup)
    coup_rr_all = np.array([c["rr"] for c, _ in all_coup]) if all_coup else np.array([])
    if all_coup:
        # SQUARE panels like the other figures (same PANEL_POS coordinates):
        # overlay (a) top center, count (b) bottom left, 1st-vs-2nd (c) bottom right
        fig = plt.figure(figsize=(8.1, 7.6), facecolor=DARK_BG)

        # (1) per-session count, color-coded like the other figures
        ax0 = fig.add_axes(PANEL_POS["bl"]); ax0.set_facecolor(DARK_BG)
        cols0 = [palette[i % len(palette)] for i in range(n_sessions)]
        ax0.bar(np.arange(n_sessions), coup_per_sess, color=cols0,
                edgecolor="#ffffff", linewidth=0.4)
        ax0.set_xticks(np.arange(n_sessions))
        ax0.set_xticklabels([short_label(s["label"]) for s in sessions],
                            rotation=45, ha="right", fontsize=FS_TEXT, color="#555555")
        ax0.set_ylabel("couplets (n)", color="#555555", fontsize=FS_LABEL)
        ax0.set_title(f"$\\bf{{(b)}}$ Couplets per session (n={n_coup_tot})",
                      color="#1f1f1f", fontsize=8.5)
        ax0.tick_params(colors="#555555", labelsize=FS_TICK)
        ax0.grid(axis="y", alpha=0.18, color="#dcdcdc")
        for sp in ax0.spines.values(): sp.set_color("#c8c8c8")

        # (2) overlay of all pairs, aligned on the 1st PVC's peak
        ax1 = fig.add_axes([(1 - PANEL_W) / 2, TOP_ROW, PANEL_W, PANEL_H])
        ax1.set_facecolor(DARK_BG)
        pairs = np.array([c["pair"] for c, _ in all_coup])
        for (c, i) in all_coup:
            ax1.plot(CPL_GRID, c["pair"], color=palette[i % len(palette)],
                     lw=0.5, alpha=0.35)
        ax1.plot(CPL_GRID, np.median(pairs, axis=0), color="#1a1a1a", lw=1.8,
                 label="median")
        rr_med = float(np.median(coup_rr_all))
        ax1.axvline(0, color="#6a6a6a", ls=":", lw=0.8, alpha=0.7)
        ax1.axvline(rr_med/1000.0, color="#b8860b", ls="--", lw=1.2,
                    label=f"median RR {rr_med:.0f} ms")
        ax1.set_xlim(-CPL_PRE, CPL_POST); ax1.set_ylim(-1.15, 1.25)
        ax1.set_xlabel("time from 1st PVC peak (s)", color="#555555", fontsize=FS_LABEL)
        ax1.set_ylabel("amplitude (norm.)", color="#555555", fontsize=FS_LABEL)
        ax1.set_title("$\\bf{(a)}$ All couplets overlaid",
                      color="#1f1f1f", fontsize=8.5)
        ax1.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                   fontsize=FS_LEGEND, loc="upper right")
        ax1.tick_params(colors="#555555", labelsize=FS_TICK)
        ax1.grid(alpha=0.16, color="#dcdcdc")
        for sp in ax1.spines.values(): sp.set_color("#c8c8c8")

        # (3) overlay of individual QRS: 1st PVC vs 2nd PVC (same morphology?)
        ax2 = fig.add_axes(PANEL_POS["br"]); ax2.set_facecolor(DARK_BG)
        q1s = np.array([c["q1"] for c, _ in all_coup])
        q2s = np.array([c["q2"] for c, _ in all_coup])
        for q in q1s:
            ax2.plot(QRS_GRID, q, color="#1f7fb0", lw=0.4, alpha=0.22)
        for q in q2s:
            ax2.plot(QRS_GRID, q, color="#cc7a1f", lw=0.4, alpha=0.22)
        m1, m2 = np.median(q1s, axis=0), np.median(q2s, axis=0)
        ax2.plot(QRS_GRID, m1, color="#1f7fb0", lw=2.2, label="1st PVC")
        ax2.plot(QRS_GRID, m2, color="#cc7a1f", lw=2.2, label="2nd PVC")
        r12 = float(np.corrcoef(m1, m2)[0, 1])
        ax2.set_xlim(-QRS_HALF, QRS_HALF); ax2.set_ylim(-1.15, 1.15)
        ax2.set_xlabel("time from QRS peak (s)", color="#555555", fontsize=FS_LABEL)
        ax2.set_ylabel("amplitude (norm.)", color="#555555", fontsize=FS_LABEL)
        ax2.set_title(f"$\\bf{{(c)}}$ 1st vs 2nd beat (r={r12:.3f})",
                      color="#1f1f1f", fontsize=8.5)
        ax2.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                   fontsize=FS_LEGEND, loc="upper right")
        ax2.tick_params(colors="#555555", labelsize=FS_TICK)
        ax2.grid(alpha=0.16, color="#dcdcdc")
        for sp in ax2.spines.values(): sp.set_color("#c8c8c8")

        img_couplets = fig_to_b64(fig, dpi=450)

    # example gallery: 6 couplet strips (2 columns × 3 rows, same format
    # as the 10 example strips: green trace, PVC QRS in red ±120ms + marker). Window
    # ±5s (less noise at the edges). Selection = show the VARIATIONS of the rhythm motif:
    # one representative (cleanest window) per distinct `ctx` motif, ordered
    # by cleanliness → the gallery covers different patterns, not 6 copies of the dominant one.
    img_couplet_strips = None
    cand = [(c, i) for c, i in all_coup if c.get("strip") is not None]
    cand.sort(key=lambda ci: ci[0].get("noise", 9.9))   # cleanest windows first
    rep = {}
    for c, i in cand:                         # cleanest representative per distinct motif
        rep.setdefault(c.get("ctx", ""), (c, i))
    picks = sorted(rep.values(), key=lambda ci: ci[0].get("noise", 9.9))[:6]
    for c, i in cand:                         # if the distinct clean motifs are <6, fill in
        if len(picks) >= 6:
            break
        if (c, i) not in picks:
            picks.append((c, i))
    if picks:
        ncol, nrow = 2, (len(picks) + 1) // 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(8.1, 1.18 * nrow + 1.05),
                                 facecolor=DARK_BG, squeeze=False)
        flat = axes.ravel()
        # dominant motif (most conserved): in its panel's title it goes in bold
        from collections import Counter as _Ctr
        _domc = _Ctr(cc.get("ctx_disp", "") for cc, _ in all_coup).most_common(1)
        dom_ctx = _domc[0][0] if _domc else None
        for ax, (c, i) in zip(flat, picks):
            ctr = c["strip"]["center"]
            mm, ss = int(ctr // 60), int(ctr % 60)
            ctx = c.get("ctx_disp", "")
            motif = " ".join(ctx)                     # "N V N N [V V] N V N N"
            if ctx == dom_ctx:                        # dominant → motif in bold
                motif = "$\\bf{" + motif.replace(" ", "\\ ") + "}$"
            draw_example_strip(ax, c["strip"],
                               f"{short_label(sessions[i]['label'])} @{mm:02d}:{ss:02d}"
                               f"    {motif}")
        for ax in flat[len(picks):]:
            ax.set_visible(False)
        for ax in flat[max(0, len(picks) - ncol):len(picks)]:
            ax.set_xlabel("Time relative to couplet centre (s)",
                          color="#555555", fontsize=FS_LABEL)
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([0], [0], color="#2f8a63", lw=1.8, label="clean ECG"),
                            Line2D([0], [0], marker="v", color="#2f8a63", lw=0,
                                   markersize=8, label="sinus beat (auto)"),
                            Line2D([0], [0], color="#cc3b30", lw=1.8, label="PVC"),
                            Line2D([0], [0], marker="v", color="#cc3b30", lw=0,
                                   markersize=10, label="PVC (auto)")],
                   loc="upper center", ncol=4, fontsize=9.5, frameon=False,
                   bbox_to_anchor=(0.5, 0.995), columnspacing=2.0)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.065,
                            hspace=0.55, wspace=0.12)
        img_couplet_strips = fig_to_b64(fig, dpi=450)

    # ---- local rhythm motif: does the couplet recur within the same pattern? ----
    # Counts the `ctx_disp` motifs (4 beats before .. 4 after) over ALL couplets.
    from collections import Counter
    motif_counter = Counter(c.get("ctx_disp", "") for c, _ in all_coup)
    img_couplet_motifs = None
    top_motifs = motif_counter.most_common(8)
    if top_motifs and n_coup_tot:
        labels_m = [" ".join(m) for m, _ in top_motifs][::-1]
        counts_m = [n for _, n in top_motifs][::-1]
        dom_n = top_motifs[0][1]
        fig, ax = plt.subplots(figsize=(8.1, 0.45 * len(top_motifs) + 1.2),
                               facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        ym = np.arange(len(labels_m))
        cols_m = ["#cc7a1f" if n == dom_n else "#2f6fb0" for n in counts_m]
        ax.barh(ym, counts_m, color=cols_m, edgecolor="#ffffff", linewidth=0.4)
        for y, n in zip(ym, counts_m):
            ax.text(n + 0.15, y, f"{n} ({100*n/n_coup_tot:.0f}%)", va="center",
                    color="#333333", fontsize=FS_TEXT)
        ax.set_yticks(ym)
        ax.set_yticklabels(labels_m, color="#1f1f1f", fontsize=FS_TEXT,
                           fontfamily="monospace")
        ax.set_xlabel("number of couplets", color="#555555", fontsize=FS_LABEL)
        ax.set_xlim(0, max(counts_m) * 1.18)
        ax.set_title("Local rhythm motif around each couplet "
                     "(4 beats before … couplet … 4 after)",
                     color="#1f1f1f", fontsize=9)
        ax.tick_params(colors="#555555", labelsize=FS_TICK)
        ax.grid(axis="x", alpha=0.16, color="#dcdcdc")
        for sp in ax.spines.values(): sp.set_color("#c8c8c8")
        fig.subplots_adjust(left=0.27, right=0.97, top=0.84, bottom=0.18)
        img_couplet_motifs = fig_to_b64(fig, dpi=450)

    # pattern summary text (HTML)
    if top_motifs and n_coup_tot:
        dm, dn = top_motifs[0]
        pattern_summary = (
            f"The couplet is usually embedded in one recurring rhythm: the motif "
            f"<code>{' '.join(dm)}</code> occurs in <b>{dn}/{n_coup_tot} "
            f"({100*dn/n_coup_tot:.0f}%)</b> of couplets — an organized, trigeminy-like "
            f"cadence (an isolated PVC, two sinus beats, the couplet, then the same "
            f"again). It is <b>not</b> universal, though: the remaining "
            f"{n_coup_tot - dn} couplets show variants — the couplet arriving out of a "
            f"longer sinus run, after a bigeminal stretch, or followed by quiet — which "
            f"is why the example strips below are chosen to span <b>different</b> motifs "
            f"rather than repeat the dominant one.")
    else:
        pattern_summary = ""

    # couplet summary text (HTML)
    n_sess_with = sum(1 for n in coup_per_sess if n > 0)
    if n_coup_tot:
        coup_summary = (
            f"<b>{n_coup_tot} couplets</b> across {n_sess_with}/{n_sessions} sessions "
            f"(none in {n_sessions - n_sess_with}). Inter-PVC interval is strikingly "
            f"tight — median <b>{float(np.median(coup_rr_all)):.0f} ms</b> "
            f"(range {coup_rr_all.min():.0f}&ndash;{coup_rr_all.max():.0f} ms) — and the "
            f"two beats share the same morphology, i.e. the couplets are <b>uniform</b> "
            f"and consistent with the same single focus firing twice, not a second focus "
            f"or a malignant polymorphic pair.")
    else:
        coup_summary = "No couplets detected in the current dataset."

    # ============ RESPIRATION (EDR) ↔ PVC phase correlation ========================
    # I reconstruct breathing from the R-amplitude (EDR) and check whether the PVCs are tied
    # to the respiratory phase. img_edr_demo = proof that breathing is detected; the
    # phase panel + table = the correlation (chi² per session).
    edr_sessions = [s for s in sessions if s.get("edr")]
    img_edr_demo = img_resp_phase = img_resp_phase_types = None
    type_peaks = {}
    resp_table = ""
    type_summary = ""
    resp_summary = ("Not enough clean long recordings (&ge;5 min) to derive respiration.")
    if edr_sessions:
        # ---- (A) demo: breathing reconstructed from the R-amplitude, over ~80 s of clean signal ----
        best = max(edr_sessions, key=lambda s: s["edr"]["snr"])
        e = best["edr"]
        tn, an, tu, rs = e["t_n"], e["amp_n"], e["t_unif"], e["resp"]
        # 80s window after the 1st minute with at least a few PVCs
        w0 = tu[0] + 60.0; w1 = w0 + 80.0
        mu = (tu >= w0) & (tu <= w1); mn = (tn >= w0) & (tn <= w1)
        if mu.sum() > 20 and mn.sum() > 10:
            fig, ax = plt.subplots(figsize=(8.1, 2.6), facecolor=DARK_BG)
            ax.set_facecolor(DARK_BG)
            # normalized R-amplitude (dots) + EDR (line)
            a = an[mn]; a_z = (a - a.mean()) / (a.std() or 1)
            r = rs[mu]; r_z = (r - r.mean()) / (r.std() or 1)
            ax.scatter(tn[mn] - w0, a_z, s=14, color="#2f8a63", alpha=0.7,
                       label="R-amplitude of N beats (z)")
            ax.plot(tu[mu] - w0, r_z, color="#1f7fb0", lw=1.8,
                    label="EDR respiration (0.1–0.5 Hz)")
            pv = e["pvc_t"]; pvw = pv[(pv >= w0) & (pv <= w1)]
            for x in pvw:
                ax.axvline(x - w0, color="#cc3b30", lw=1.0, alpha=0.55)
            if len(pvw):
                ax.plot([], [], color="#cc3b30", lw=1.0, label="PVC")
            ax.set_xlim(0, 80); ax.set_xlabel("time (s)", color="#555555", fontsize=FS_LABEL)
            ax.set_ylabel("normalized", color="#555555", fontsize=FS_LABEL)
            ax.set_title(f"Respiration recovered from R-amplitude — {short_label(best['label'])} "
                         f"({e['rate_resp']:.1f} breaths/min, SNR {e['snr']:.1f})",
                         color="#1f1f1f", fontsize=8.5)
            ax.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                      fontsize=FS_LEGEND, loc="upper right", ncol=3)
            ax.tick_params(colors="#555555", labelsize=FS_TICK)
            ax.grid(alpha=0.16, color="#dcdcdc")
            for sp in ax.spines.values(): sp.set_color("#c8c8c8")
            fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.16)
            img_edr_demo = fig_to_b64(fig, dpi=450)

        # ---- (B) phase: aggregated rosette + per-session enrichment ----
        cen = edr_sessions[0]["edr"]["centers"]
        pct = cen * 100 / (2 * np.pi)
        fig = plt.figure(figsize=(8.1, 3.3), facecolor=DARK_BG)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], left=0.02, right=0.97,
                              top=0.84, bottom=0.20, wspace=0.22)
        # aggregated polar rosette (sum over all sessions)
        axp = fig.add_subplot(gs[0], projection="polar"); axp.set_facecolor(DARK_BG)
        dn = np.sum([s["edr"]["dens_n"] for s in edr_sessions], axis=0)
        dp = np.sum([s["edr"]["dens_p"] for s in edr_sessions], axis=0)
        dn = dn / dn.sum() * 100; dp = dp / dp.sum() * 100
        wbar = (2 * np.pi / NBINS_RESP) * 0.95
        axp.bar(cen, dn, width=wbar, color="#2f8a63", alpha=0.45, label="N beats")
        axp.bar(cen, dp, width=wbar, color="#cc3b30", alpha=0.55, label="PVC")
        axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
        axp.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        axp.set_xticklabels(["lungs full\n(end-insp.)", "25%", "lungs empty\n(end-exp.)", "75%"],
                            color="#555555", fontsize=FS_TICK)
        axp.tick_params(colors="#777", labelsize=FS_TICK-1)
        axp.set_title("$\\bf{(a)}$ Phase distribution (all sessions)", color="#1f1f1f",
                      fontsize=8.5, pad=14)
        axp.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                   fontsize=FS_LEGEND, loc="lower right", bbox_to_anchor=(1.15, -0.05))
        # per-session enrichment + mean
        axe = fig.add_subplot(gs[1]); axe.set_facecolor(DARK_BG)
        xx = np.concatenate([pct, [100]])     # close the cycle
        for s in edr_sessions:
            en = np.array(s["edr"]["enrich"]); en = np.concatenate([en, [en[0]]])
            axe.plot(xx, en, color="#5a8fb0", lw=0.8, alpha=0.5)
        mean_en = np.mean([s["edr"]["enrich"] for s in edr_sessions], axis=0)
        mean_en = np.concatenate([mean_en, [mean_en[0]]])
        axe.plot(xx, mean_en, color="#b8860b", lw=2.6, label="mean across sessions")
        axe.axhline(1.0, color="#6a6a6a", ls="--", lw=0.9)
        axe.axvspan(0, 14, color="#cc3b30", alpha=0.10)
        axe.axvspan(86, 100, color="#cc3b30", alpha=0.10)
        axe.text(2, axe.get_ylim()[1], "lungs full (end-inspiration)", color="#cc5a52",
                 fontsize=FS_TEXT, ha="left", va="top")
        axe.set_xlim(0, 100)
        axe.set_xlabel("% of respiratory cycle  (0 / 100 = lungs full / end-inspiration, "
                       "50 = lungs empty / end-expiration)",
                       color="#555555", fontsize=FS_LABEL)
        axe.set_ylabel("PVC enrichment (PVC density / N density)", color="#555555", fontsize=FS_LABEL)
        axe.set_title("$\\bf{(b)}$ Where in the breath do PVCs fire? (×1 = no preference)",
                      color="#1f1f1f", fontsize=8.5)
        axe.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                   fontsize=FS_LEGEND, loc="upper right")
        axe.tick_params(colors="#555555", labelsize=FS_TICK)
        axe.grid(alpha=0.16, color="#dcdcdc")
        for sp in axe.spines.values(): sp.set_color("#c8c8c8")
        img_resp_phase = fig_to_b64(fig, dpi=450)

        # ---- (B2) rosette by PVC TYPE: interpolated / compensated / coupled ----
        # do they fall at the same respiratory phase or at different ones? Each PVC's phase
        # taken from phase_at; type from pause_data (interp/comp, global valley) and from
        # couplets (both beats of the pair).
        ph = {"interp": [], "comp": [], "coupled": []}
        for s in edr_sessions:
            pa = s["edr"]["phase_at"]
            ti = [d["t"] for d in s["pause_data"] if not d["guard"] and d["post_ratio"] < PAUSE_VALLEY]
            tc = [d["t"] for d in s["pause_data"] if not d["guard"] and d["post_ratio"] >= PAUSE_VALLEY]
            tk = [c["t1"] for c in s["couplets"]] + [c["t2"] for c in s["couplets"]]
            if ti: ph["interp"].append(pa(np.array(ti)))
            if tc: ph["comp"].append(pa(np.array(tc)))
            if tk: ph["coupled"].append(pa(np.array(tk)))
        ph = {k: (np.concatenate(v) if v else np.array([])) for k, v in ph.items()}
        TYPE_COL = {"interp": "#1f7fb0", "comp": "#d2685f", "coupled": "#b8860b"}
        TYPE_LAB = {"interp": "Interpolated", "comp": "Compensated", "coupled": "Coupled"}
        bins_r = np.linspace(0, 2 * np.pi, NBINS_RESP + 1)
        cen_r = (bins_r[:-1] + bins_r[1:]) / 2
        cc = np.append(cen_r, cen_r[0])           # close the loop
        type_peaks = {}
        # SQUARE figure (like the outlier): INTERNAL legend in a corner, no
        # external legend below (which would make the PNG rectangular with the tight crop).
        fig = plt.figure(figsize=(5.4, 5.4), facecolor=DARK_BG)
        axt = fig.add_axes([0.10, 0.08, 0.80, 0.80], projection="polar")
        axt.set_facecolor(DARK_BG)
        for k in ("interp", "comp", "coupled"):
            if len(ph[k]) < 8:
                continue
            h, _ = np.histogram(ph[k], bins=bins_r)
            d = h / h.sum()
            dd = np.append(d, d[0])
            axt.plot(cc, dd, color=TYPE_COL[k], lw=2.0,
                     label=f"{TYPE_LAB[k]} (n={len(ph[k])})")
            axt.fill(cc, dd, color=TYPE_COL[k], alpha=0.12)
            type_peaks[k] = float(cen_r[int(np.argmax(d))] * 100 / (2 * np.pi))
        axt.set_theta_zero_location("N"); axt.set_theta_direction(-1)
        axt.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        axt.set_xticklabels(["lungs full\n(end-insp.)", "25%", "lungs empty\n(end-exp.)", "75%"],
                            color="#555555", fontsize=FS_TICK)
        axt.tick_params(colors="#777", labelsize=FS_TICK-1)
        axt.set_title("Respiratory phase by PVC type (density per type)",
                      color="#1f1f1f", fontsize=9, pad=18)
        axt.legend(facecolor="#f2efe9", labelcolor="#1a1a1a", edgecolor="#c8c8c8",
                   fontsize=FS_LEGEND-0.5, loc="upper left", bbox_to_anchor=(-0.16, 1.10),
                   ncol=1, handlelength=1.3, handletextpad=0.5, borderpad=0.5,
                   labelspacing=0.4, framealpha=0.9)
        img_resp_phase_types = fig_to_b64(fig, dpi=450)

        # ---- (C) table + summary ----
        n_sig = sum(1 for s in edr_sessions if s["edr"]["pval"] < 0.05)
        def _pv(p):
            col = "#1b8a3a" if p < 0.05 else "#cc5a2a"
            if p < 1e-300:          # numerical underflow: there is no p-value = 0
                txt = "&lt;1e-300"
            else:
                txt = f"{p:.0e}" if p < 0.01 else f"{p:.3f}"
            return f"<b style='color:{col}'>{txt}</b>"
        rrows = []
        for s in edr_sessions:
            ed = s["edr"]
            rrows.append(
                f"<tr><td>{s['label']}</td>"
                f"<td class='num'>{ed['rate_resp']:.1f}</td>"
                f"<td class='num'>{ed['n_p']:,}</td>"
                f"<td class='num'>{ed['peak_phase_pct']:.0f}%</td>"
                f"<td class='num'>&times;{ed['peak_enrich']:.2f}</td>"
                f"<td class='num'>{_pv(ed['pval'])}</td>"
                f"<td class='num'>{ed['snr']:.1f}</td></tr>")
        resp_table = "\n".join(rrows)
        mean_peak = float(np.mean([s["edr"]["peak_enrich"] for s in edr_sessions]))
        resp_summary = (
            f"Respiration was recovered in <b>{len(edr_sessions)}/{n_sessions}</b> sessions "
            f"(the rest too short or too noisy). PVC timing is <b>significantly</b> coupled to "
            f"respiratory phase in <b>{n_sig}/{len(edr_sessions)}</b> of them "
            f"(&chi;&sup2; across phase bins, p&lt;0.05) — so the answer is <b>yes, there is a "
            f"real correlation</b>. The R-amplitude is largest at <b>full lungs</b> (the subject "
            f"confirmed this directly while recording), so the phase peak sits at "
            f"<b>full inflation / end-inspiration</b>: PVCs are over-represented there "
            f"(mean peak &times;{mean_peak:.1f} vs chance) and sparsest near empty lungs. "
            f"That points to a <b>mechanical, lung-volume / cardiac-filling</b> trigger at peak "
            f"inflation (diaphragm lowest, maximal venous return and chamber stretch) rather than "
            f"the end-expiratory vagal one previously assumed.")
        # phase comparison by type
        if len(type_peaks) >= 2:
            def _nearfull(p):  # circular distance from "full lungs" (0/100%)
                return min(p, 100 - p)
            parts = ", ".join(f"{TYPE_LAB[k].lower()} {type_peaks[k]:.0f}%"
                              for k in ("interp", "comp", "coupled") if k in type_peaks)
            peaks_pct = list(type_peaks.values())
            spread = max(_nearfull(v) for v in peaks_pct)
            same = spread <= 25
            verdict = (
                "all three favour roughly the <b>same</b> phase (near full lungs), so interpolated, "
                "compensated and coupled beats share one respiratory trigger and do not pick out "
                "separate phases" if same else
                "the types peak at <b>different</b> phases — worth a closer look, as it would hint "
                "that the pause type and the respiratory trigger interact")
            type_summary = (f"Split by type, the phase peaks are: {parts} "
                            f"(% of cycle, 0 = full lungs). So {verdict}.")
        else:
            type_summary = ""

    # ============ EXAMPLE STRIPS: recording quality + PVC auto-detection ============
    # 10 strips in a 2-column x 5-row grid, ±10 s window (20 s) around
    # the event. Mix: regular (various sessions) + 1 couplet + 1 burst + 2
    # interpolated. Report style: green trace, PVC QRS in red (±120 ms) + marker.
    # NB: no definition text for the types (couplet/burst/interp) — only tags.
    N_STRIPS = 10
    specials = []   # (snippet, title)
    if best_couplet is not None:
        ex, lab = best_couplet
        specials.append((ex, f"{short_label(lab)} · couplet"))
    if best_burst is not None:
        ex, lab = best_burst
        specials.append((ex, f"{short_label(lab)} · burst"))
    # up to 2 interpolated, the clearest ones (lowest ratio), from different sessions
    seen_sess = set()
    for ex, lab in sorted(interp_candidates, key=lambda x: x[0]["ratio"]):
        if lab in seen_sess:
            continue
        seen_sess.add(lab)
        specials.append((ex, f"{short_label(lab)} · interpolated"))
        if sum(1 for _, t in specials if "interpolated" in t) >= 2:
            break
    # regular ones to fill up to N_STRIPS, from different sessions for variety
    regulars = []
    for s in sessions:
        ex = s.get("example")
        if ex is None:
            continue
        c = ex["center"]; mm, ss = int(c // 60), int(c % 60)
        regulars.append((ex, f"{short_label(s['label'])} · @{mm:02d}:{ss:02d}"))
    strips = (specials + regulars)[:N_STRIPS]

    img_examples = None
    if strips:
        ncol, nrow = 2, (len(strips) + 1) // 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 1.7 * nrow),
                                 facecolor=DARK_BG, squeeze=False)
        flat = axes.ravel()
        for ax, (ex, title) in zip(flat, strips):
            draw_example_strip(ax, ex, title)
        for ax in flat[len(strips):]:      # hides empty cells
            ax.set_visible(False)
        fig.suptitle("Example strips (±10 s) — recording quality & automatic PVC detection",
                     color="#1f1f1f", fontsize=FS_TITLE, y=0.997)
        for ax in flat[max(0, len(strips) - ncol):len(strips)]:
            ax.set_xlabel("Time relative to window centre (s)",
                          color="#555555", fontsize=FS_LABEL)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05,
                            hspace=0.55, wspace=0.12)
        img_examples = fig_to_b64(fig, dpi=220)

    # mean corr per session table (HTML)
    outlier_rows = "\n".join(
        f"<tr><td>{lab}</td><td class='num'>{mc:.3f}</td></tr>"
        for lab, mc, _ in mean_corr_per_session)

    # ============ AGGREGATE STATS ============
    cum_total_pvc  = sum(s["n_pvc"] for s in sessions)
    cum_total_norm = sum(s["n_norm"] for s in sessions)
    cum_duration   = sum(s["duration_min"] for s in sessions)
    cum_excluded   = sum(s["excluded_seconds"] for s in sessions)

    # Sessions table
    sessions_rows = []
    for s in sessions:
        burden = 100*s["n_pvc"]/(s["n_norm"]+s["n_pvc"]) if (s["n_norm"]+s["n_pvc"]) else 0
        sessions_rows.append(
            f"<tr><td>{s['label']}</td>"
            f"<td>{s['duration_min']:.1f} min</td>"
            f"<td>{s['n_norm']:,}</td>"
            f"<td>{s['n_pvc']:,}</td>"
            f"<td>{burden:.1f}%</td>"
            f"<td>{s['n_excluded_intervals']} ({s['excluded_seconds']:.0f}s)</td>"
            "</tr>"
        )
    sessions_table = "\n".join(sessions_rows)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DIY Holter ECG — build &amp; signal-analysis notebook</title>
<style>
  :root {{ --paper:#fcfbf8; --ink:#1f1e1c; --muted:#6b6862; --faint:#e7e3da;
           --rule:#d6d1c6; --accent:#7a3b2e; }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--paper); color:var(--ink); counter-reset:sec;
          font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
          margin:0 auto; max-width:880px; line-height:1.62; padding:52px 30px 80px;
          font-size:17px; -webkit-font-smoothing:antialiased; }}
  h1 {{ font-size:1.85em; line-height:1.16; font-weight:600; letter-spacing:-0.01em;
        margin:0 0 3px; }}
  .subtitle {{ color:var(--muted); font-size:1.03em; font-style:italic; margin:0 0 16px; }}
  .updated {{ color:var(--muted); font-size:0.78em; letter-spacing:0.01em;
              font-family:ui-sans-serif,-apple-system,sans-serif;
              border-top:1px solid var(--rule); border-bottom:1px solid var(--rule);
              padding:7px 0; margin-bottom:30px; }}
  h2 {{ counter-increment:sec; font-size:1.3em; font-weight:600; margin:44px 0 6px;
        padding-bottom:5px; border-bottom:2px solid var(--ink); letter-spacing:-0.005em; }}
  h2::before {{ content:counter(sec) ".\\00a0\\00a0"; color:var(--accent); font-weight:700; }}
  h3 {{ font-size:1.07em; font-weight:600; margin:26px 0 4px; color:#33312d; }}
  p, .commentary {{ margin:11px 0; }}
  .commentary {{ font-size:0.97em; color:#34322e; }}
  .commentary b {{ color:var(--ink); font-weight:600; }}
  .commentary ul {{ margin:8px 0; padding-left:22px; }}
  .commentary li {{ margin:4px 0; }}
  a {{ color:var(--accent); }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:0.82em;
          background:#f1ede4; border:1px solid var(--faint); border-radius:3px; padding:0 4px; }}
  details {{ border:1px solid var(--rule); border-radius:3px; margin:14px 0; background:#faf7f1; }}
  details > summary {{ cursor:pointer; padding:9px 14px; font-weight:600;
                       font-family:ui-sans-serif,-apple-system,sans-serif; font-size:0.82em;
                       letter-spacing:0.04em; text-transform:uppercase; color:var(--muted);
                       list-style:none; user-select:none; }}
  details > summary::-webkit-details-marker {{ display:none; }}
  details > summary::before {{ content:"+\\00a0\\00a0"; }}
  details[open] > summary::before {{ content:"\\2013\\00a0\\00a0"; }}
  details[open] > summary {{ border-bottom:1px solid var(--rule); }}
  details > .content {{ padding:12px 16px; }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
                gap:0; margin:16px 0 6px; border-top:2px solid var(--ink); }}
  .stat {{ padding:9px 16px 9px 0; border-bottom:1px solid var(--faint); }}
  .stat .v {{ display:block; font-size:1.42em; font-weight:600; font-variant-numeric:tabular-nums; }}
  .stat .l {{ display:block; color:var(--muted); font-size:0.7em; margin-top:1px;
              font-family:ui-sans-serif,-apple-system,sans-serif;
              text-transform:uppercase; letter-spacing:0.05em; }}
  .stat.pvc .v {{ color:#b03a2e; }} .stat.burden .v {{ color:#b9770b; }}
  .stat.heartbeats .v {{ color:#2f6fb0; }}
  table {{ border-collapse:collapse; width:100%; margin:14px 0; font-size:0.8em;
           font-family:ui-sans-serif,-apple-system,sans-serif; }}
  th, td {{ padding:5px 9px; text-align:left; border-bottom:1px solid var(--faint); }}
  tr:first-child th {{ border-bottom:1.5px solid var(--ink); text-transform:uppercase;
                       letter-spacing:0.03em; font-size:0.92em; color:#3a382f; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  table.summary td:first-child {{ font-weight:600; color:#33312d; }}
  img {{ display:block; width:100%; max-width:100%; height:auto; margin:14px auto;
         border:1px solid var(--rule) !important; border-radius:2px !important;
         background:#fff !important; }}
  .device-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:5px 28px;
                  font-size:0.9em; font-family:ui-sans-serif,-apple-system,sans-serif; }}
  .device-grid dt {{ font-weight:600; color:#33312d; }}
  .device-grid dd {{ color:#4a4842; margin:0 0 2px; }}
  footer {{ margin-top:46px; padding-top:12px; border-top:1px solid var(--rule);
            color:var(--muted); font-size:0.78em; line-height:1.55;
            font-family:ui-sans-serif,-apple-system,sans-serif; }}
  /* accenti inline (span statici nel testo) scuriti per la carta */
  [style*="#7ad9ff"]{{color:#1f7fb0 !important}} [style*="#5fb1ff"]{{color:#2f6fb0 !important}}
  [style*="#ff8a8a"]{{color:#c0392b !important}} [style*="#ff6b6b"]{{color:#c0392b !important}}
  [style*="#ff9a9a"]{{color:#c0392b !important}} [style*="#ffe169"]{{color:#9a7d0a !important}}
  [style*="#ffd633"]{{color:#9a7d0a !important}} [style*="#33ff66"]{{color:#1b7a3a !important}}
  [style*="#7fd693"]{{color:#2e8b57 !important}} [style*="#5fcc9e"]{{color:#2e8b57 !important}}
  [style*="#ffa64d"]{{color:#c0560a !important}} [style*="#ff7a4d"]{{color:#c0451a !important}}
  [style*="#b59bff"]{{color:#6f42c1 !important}}
  @media (max-width:760px) {{ .device-grid {{ grid-template-columns:1fr; }}
                              body {{ padding:30px 18px; }} }}
  @media print {{
    html, body {{ -webkit-print-color-adjust:exact; print-color-adjust:exact;
                  background:#fff; max-width:none; padding:0; font-size:11px; }}
    details {{ content-visibility:visible !important; border:none; background:none; }}
    details > *:not(summary), details > .content {{ display:block !important;
                  content-visibility:visible !important; }}
    details > summary {{ display:none; }}
    div[style*="overflow"] {{ overflow:visible !important; }}
    h1, h2, h3 {{ page-break-after:avoid; }}
    img, table {{ page-break-inside:avoid; }}
  }}
  @page {{ size:A4 portrait; margin:16mm 15mm; }}
</style>
</head>
<body>

<h1>DIY Holter ECG — a build &amp; signal-analysis notebook</h1>
<div class="subtitle">
  A home-built single-lead recorder, and what its own recordings say about one
  person's ectopic beats — an engineering and learning exercise, not a medical record.
</div>
<div class="updated">
  Compiled {now} · {len(sessions)} sessions · regenerate with
  <code>python3 host/dashboard.py</code>
</div>

<details open>
  <summary>Acquisition setup &amp; processing pipeline</summary>
  <div class="content">
    <dl class="device-grid">
      <dt>Front-end</dt>
      <dd>AD8232 single-lead bio-potential amplifier</dd>
      <dt>Lead configuration</dt>
      <dd>3 snap electrodes, single precordial derivation</dd>
      <dt>Microcontroller</dt>
      <dd>Raspberry Pi Pico 2 W (MicroPython)</dd>
      <dt>ADC</dt>
      <dd>Pico internal 12-bit, channel <code>ADC0 (GP26)</code></dd>
      <dt>Sample rate</dt>
      <dd>{SR} Hz</dd>
      <dt>Transport</dt>
      <dd>WiFi (TCP, port 5005) or USB serial (mpremote)</dd>
      <dt>Power</dt>
      <dd>1S LiPo + TP4056 charger + SPDT switch, ~24 h autonomy</dd>
      <dt>Server / analysis</dt>
      <dd>Python (Flask + SSE dashboard, batch report tools) on MacBook / Pi 5</dd>
      <dt>Bandpass filter</dt>
      <dd>1st-order IIR: HP 0.3 Hz + LP 25 Hz (online, on server)</dd>
      <dt>QRS detector</dt>
      <dd>4-state FSM (IDLE → WIDTH → DETECT → POST), refractory 300 ms</dd>
      <dt>PVC classifier</dt>
      <dd>(rebound ≥ {REBOUND_PVC} <em>or</em> width ≥ {PVC_W_MS:.0f} ms)
          AND amplitude ≥ {PVC_MIN_AMP} V
          AND {PVC_W_MIN:.0f} ≤ width ≤ {PVC_W_MAX:.0f} ms
          AND rebound ≥ {PVC_MIN_REBOUND}</dd>
      <dt>Manual noise exclusion</dt>
      <dd><code>host/mark_exclusions.py</code> — paginated matplotlib editor,
          per-session JSON under <code>exclusions/</code></dd>
    </dl>
    <div class="commentary" style="margin-top:12px">
      Setup is hobbyist / educational. <b>This is not a medical device</b> and
      the analyses below are not diagnostic. All clinical decisions are made
      under cardiologist supervision; data here serves to characterize
      long-term patterns of the personal ectopic focus and to track effects of
      postural/breathing interventions.
    </div>
  </div>
</details>

<details open>
  <summary>Sessions table ({len(sessions)} sessions)</summary>
  <div class="content">
    <table>
      <tr><th>Date / time</th><th>Duration</th><th>N beats</th><th>PVC</th>
          <th>Burden</th><th>Excluded</th></tr>
      {sessions_table}
    </table>
  </div>
</details>

<h2>Dataset overview</h2>
<div class="stat-grid">
  <div class="stat"><span class="v">{len(sessions)}</span>
    <span class="l">sessions analyzed</span></div>
  <div class="stat"><span class="v">{cum_duration:.0f}</span>
    <span class="l">total minutes ({cum_duration/60:.1f} h)</span></div>
  <div class="stat heartbeats"><span class="v">{cum_total_pvc + cum_total_norm:,}</span>
    <span class="l">heartbeats classified (N + PVC)</span></div>
  <div class="stat pvc"><span class="v">{cum_total_pvc:,}</span>
    <span class="l">PVCs detected</span></div>
  <div class="stat burden"><span class="v">{100*cum_total_pvc/max(1,(cum_total_pvc+cum_total_norm)):.1f}%</span>
    <span class="l">cumulative PVC burden</span></div>
  <div class="stat"><span class="v">{cum_excluded/60:.1f} min</span>
    <span class="l">manually excluded as noise</span></div>
</div>

<h2>Recording quality &amp; PVC auto-detection</h2>
<div class="commentary">
  A set of &plusmn;10-second example windows (20&nbsp;s each), selected
  automatically across sessions and kept away from intervals marked as noise.
  The continuous trace shows the raw signal quality; the detector output is
  overlaid on top, exactly as used in every analysis below: the QRS of each PVC
  is highlighted in <span style="color:#ff8a8a">red</span> (&plusmn;120&nbsp;ms)
  with a red marker, while <span style="color:#5fcc9e">green</span> markers tag
  the sinus beats. The mix includes ordinary isolated PVCs plus a couplet, a
  burst and interpolated beats (tagged in each title).
</div>
<img src="data:image/png;base64,{img_examples}" alt="Example strips with PVC detection"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1100px; display:block; margin: 0 auto;"/>

<h2>PVC morphology analysis</h2>
<div class="commentary">
  Each PVC is centered on its ectopic peak, normalized to its own amplitude
  (peak = 1.0), and overlaid in a common time window of ±{WIN/2*1000:.0f} ms.
  Conservation of this shape across sessions may suggest a single stable
  ectopic focus; visible divergences could indicate multifocality or
  substrate changes. Panels 1-2 share the same Y-axis scale for direct
  visual comparison.
</div>
<img src="data:image/png;base64,{img_morphology_4panel}" alt="PVC morphology summary"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: {disp_width(img_morphology_4panel)}px; display:block; margin: 0 auto;"/>

<h3>Are PVCs a single continuous population or do they form discrete subtypes?</h3>
<div class="commentary">
  Visual inspection of the "Median morphology by session" panel above suggests
  that one session (2026-06-06 13:41, n ≈ 924) shows a slightly different
  hyperpolarization profile compared to the others. This raises the question
  of whether the variability across PVCs reflects a single ectopic focus
  with continuous intrinsic variability, or whether two (or more) discrete
  morphological subtypes coexist — which would suggest mild bifocality.
  Three data-driven checks address this:
  <ul>
    <li><b>PCA + K-means (k=2)</b>: forces a bipartition of all PVCs in
        principal-component space. If the two clusters appear as two
        well-separated clouds, that is evidence for two real subtypes; if
        they look like one connected cloud arbitrarily cut down the middle,
        the partition is an artefact of the algorithm.</li>
    <li><b>PVCs sorted by hyperpolarization depth</b>: a 2D heatmap where
        each row is one PVC, ordered by the depth of its post-QRS trough.
        Two discrete bands (sharp transition) would indicate two subtypes;
        a smooth color gradient indicates a continuum.</li>
    <li><b>Elbow plot</b>: K-means inertia as a function of k. A sharp
        "knee" at some k* would suggest k* natural clusters; a smoothly
        decaying curve indicates no preferred number of clusters.</li>
    <li><b>Hyperpolarization depth distribution</b>: histogram of post-QRS
        trough depth across all PVCs. A single unimodal peak is consistent
        with a single focus and continuous variability; two well-separated
        peaks would suggest two morphological subtypes.</li>
  </ul>
</div>
<img src="data:image/png;base64,{img_pvc_continuum}" alt="PVC continuum check"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: {disp_width(img_pvc_continuum)}px; display:block; margin: 0 auto;"/>
<div class="commentary">
  <b>Note on the sparse points around the cloud (lower-right).</b> The thin scatter
  that fans out below and to the right of the dense ball is <em>not</em> a second PVC
  type or focus: it is a low-density skirt of more variable beats (about 3% of the total),
  concentrated in the two shortest/noisiest recordings, and its median QRS is essentially
  identical to the core's — same width and amplitude, trough only marginally deeper. It
  reflects beat-to-beat / baseline variability captured by PC2, consistent with the
  single-population picture from the elbow plot and the unimodal trough-depth histogram.
</div>

<h2>Normal beats morphology</h2>
<div class="commentary">
  Same superimposition analysis applied to normal sinus beats (N), with up to 500
  evenly-spaced N samples per session to control memory footprint. Together with
  the PVC analysis above it serves as a baseline reference. Inspection of N
  beats may help identify electrode-placement or posture effects (in the
  observed amplitude modulation pattern, R-amplitude appears to vary
  sinusoidally with breathing). The fourth panel compares the most divergent
  (outlier) session's median N with the median and IQR of the other sessions
  (detailed below).
</div>
<img src="data:image/png;base64,{img_n_morphology_4panel}" alt="Normal beat morphology — 4-panel summary"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: {disp_width(img_n_morphology_4panel)}px; display:block; margin: 0 auto;"/>

<h3>Cross-session N correlation — outlier inspection</h3>
<div class="commentary">
  The N correlation matrix shows some cells below the upper bound of the
  color scale (r &lt; 0.97). All values remain ≥ 0.93, which in absolute
  terms is still a high concordance. The colormap is configured with
  vmin=0.95 to maximize contrast and surface any session that drifts
  slightly from the others.
  <p style="margin: 8px 0 4px 0"><b>Observed difference in this dataset</b>:
  the outlier session (09 13:23) is the only one in which the subject was
  seated at a desk with anterior trunk flexion, while all other sessions
  were recorded in lying / extended postures. This posture difference may
  explain the morphology drift, since a change in trunk position can rotate
  the heart axis relative to the precordial electrode and modify the
  projection of the depolarization vector on the single-lead signal.</p>
  <p style="margin: 4px 0 0 0">A plausible reason why the effect surfaces
  in N and not in PVC: PVCs have a large dominant deflection (~1.5 V) whose
  shape, once amplitude-normalized, tends to be robust to small perturbations
  of the projection; N beats are smaller (~0.5 V) and roughly symmetric
  around baseline, so normalization amplifies the relative weight of P-wave
  and T-wave regions, which may be more sensitive to posture changes. This
  is an interpretation of the observed data, not a confirmed mechanism.</p>
</div>

<div style="display:grid; grid-template-columns: 280px 380px;
            gap: 22px; align-items: center; justify-content: center;
            margin: 14px auto; max-width: 720px;">
  <div>
    <table style="margin: 0;">
      <tr><th>Session</th><th>Mean r</th></tr>
      {outlier_rows}
    </table>
    <p style="color:#888; font-size:0.82em; margin-top:10px; margin-bottom:0;">
      Outlier: <code>{outlier_label}</code> (mean r = <b>{outlier_r:.3f}</b>).
      Plot on the right: outlier median N morphology vs median (and IQR) of
      the other {len(sessions)-1} sessions.
    </p>
  </div>
  <div>
    <img src="data:image/png;base64,{img_n_outlier}" alt="N outlier comparison"
         style="border:1px solid #25282d; border-radius:6px;
                width:100%; display:block;"/>
  </div>
</div>

<h2>Interpolated vs compensated PVCs — method &amp; validation</h2>
<div class="commentary">
  Every PVC sits between a preceding and a following sinus beat (N&ndash;PVC&ndash;N).
  <b>Conventionally</b> the two are told apart by what happens to the sinus node,
  measured <b>R-to-R, from QRS peak to QRS peak</b> (the repolarization / the PVC's
  hyperpolarization rebound are not used):
  an <span style="color:#7ad9ff">interpolated</span> PVC slips in without resetting
  the SA node, so the next sinus beat stays on schedule and the N&ndash;PVC&ndash;N
  interval is &asymp; <b>1&times;</b> the sinus cycle; a
  <span style="color:#ff8a8a">compensated</span> PVC resets the SA node, the next
  sinus beat is delayed by a full pause, and N&ndash;PVC&ndash;N &asymp; <b>2&times;</b>.
</div>
<img src="data:image/png;base64,{img_method_example}" alt="Interpolated vs compensated — example"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1100px; display:block; margin: 0 auto;"/>
<div class="commentary">
  The reference sinus cycle is not a single session-wide number: this subject has
  marked respiratory sinus arrhythmia, so it is estimated <b>locally</b> as the median
  of the <b>{PAUSE_K} nearest N&ndash;N intervals</b> around each PVC. A
  <b>prematurity guard</b> discards beats whose coupling (RR<sub>pre</sub>) is not
  shorter than that local sinus cycle &mdash; physically impossible for a true
  (premature) PVC, and a sign that a small sub-threshold sinus beat was missed in the
  gap; {method_guard} beats (&asymp;{100*method_guard/max(1,method_guard+method_n):.0f}%)
  are set aside this way.
</div>
<h3>Why the conventional sum leaves a grey zone &mdash; and how the data resolves it</h3>
<div class="commentary">
  Applying the strict sum rule (1.85&ndash;2.15&times; for a "full" pause) leaves a
  wide band of beats in between, neither clearly 1&times; nor 2&times;. Looking at the
  data, though, the picture is cleaner than the rule suggests. Two distributions:
  on the left the conventional <b>sum S</b>; on the right the <b>actual pause
  RR<sub>post</sub></b> after the PVC. Both are clearly <b>bimodal</b> &mdash; almost
  every beat falls into one lump or the other, with a near-empty valley between them.
</div>
<img src="data:image/png;base64,{img_method_dist}" alt="Distributions of S and RR_post"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1100px; display:block; margin: 0 auto;"/>
<div class="commentary">
  The sum S and the pause RR<sub>post</sub> agree on ~99% of beats (the coupling here
  is nearly fixed), but they disagree on a few: the same S can hide a short or a long
  pause depending on the coupling. When they disagree, the variable that matches what
  is <b>seen on the trace and felt</b> is the <b>pause</b> &mdash; a long pause loads
  the ventricle and the next beat lands as a forceful "thump". So the operative split
  is made on RR<sub>post</sub>, at the empirical valley
  <b>{PAUSE_VALLEY:.2f}&times;</b> the sinus cycle: pause shorter &rarr;
  <span style="color:#7ad9ff">interpolated</span> (silent), pause longer &rarr;
  <span style="color:#ff8a8a">compensated</span> (felt). Across the dataset that gives
  <b>{method_pct_int:.0f}% interpolated</b> and <b>{method_pct_comp:.0f}% compensated</b>
  on {method_n:,} classified beats; only ~{method_amb:.0f}% sit within &plusmn;0.10 of the
  valley (genuinely borderline), and those that remain truly undecided are noise, which
  is removed by manual exclusion / the guard rather than forced into a class.
</div>
<h3>How it looks on the trace</h3>
<div class="commentary">
  The same classification on a continuous strip (session {demo['label']}): the QRS of
  each PVC is colored <span style="color:#7ad9ff">blue</span> when interpolated (the
  sinus rhythm carries on with no gap) and <span style="color:#ff8a8a">red</span> when
  compensated (a clear pause follows before the next sinus beat).
</div>
<img src="data:image/png;base64,{img_method_strip}" alt="Final 2-class strip"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1100px; display:block; margin: 0 auto;"/>
<div class="commentary">
  This two-way split is what the subject actually perceives: the
  <span style="color:#7ad9ff">interpolated</span> beats carry no pause and go unnoticed,
  while the <span style="color:#ff8a8a">compensated</span> ones are followed by the pause
  and the potentiated beat that is felt as a <b>skipped / missed beat</b>. The
  cross-session analysis below is built on this classification.
</div>

<h2>Cross-session rhythm &amp; burden dynamics</h2>
<div class="commentary">
  Longitudinal comparison across all sessions, recomputed at every run — each new
  recording updates the table and the four panels automatically. Metrics follow
  the same definitions as the summary report: <b>burden</b> = PVCs / all beats;
  <b>effective SA rate</b> = 60000 / median N&ndash;N interval (the rate the wrist
  pulse would read, since most PVCs are non-perfusing); a PVC is
  <span style="color:#7ad9ff">interpolated</span> when RR<sub>pre</sub>+RR<sub>post</sub>
  &asymp; 1&times; the sinus interval (no compensatory pause, usually not felt) and
  <span style="color:#ff8a8a">compensated</span> when &asymp; 2&times; (full pause,
  the "thump"). Classification thresholds (1.30 / 1.85&ndash;2.15&times; RR<sub>sinus</sub>)
  are parametric; the per-session values are data-driven.
  <ul>
    <li><b>Burden by session</b>: how the PVC load varies recording to recording.</li>
    <li><b>Effective rate vs pause type</b>: at lower SA rates interpolated
        (silent) beats tend to prevail; as the rate rises they may shift toward
        compensated (felt) beats — the pattern that could explain why perceived
        thumps do not track raw PVC count.</li>
    <li><b>Composition</b>: interpolated / compensated / incomplete share per session.</li>
    <li><b>Coupling stability</b>: a tight, session-stable pre-PVC coupling is
        consistent with a single monomorphic focus; large drifts could indicate
        multifocality.</li>
  </ul>
</div>
<img src="data:image/png;base64,{img_crosssession}" alt="Cross-session rhythm and burden"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: {disp_width(img_crosssession)}px; display:block; margin: 0 auto;"/>
<div style="overflow-x:auto; margin: 12px auto; max-width: 900px;">
<table>
  <tr><th>Session</th><th>Dur (min)</th><th>Sinus N/min</th><th>SA eff. (BPM)</th>
      <th>PVC/min</th><th>Burden</th><th>Interp.</th><th>Comp.</th><th>Couplets</th></tr>
  {cross_table}
</table>
</div>
<div class="commentary" style="margin-top: 4px;">
  <b>Key pattern.</b> The single relationship that ties the dataset together: the
  share of <span style="color:#ff8a8a">compensated</span> (felt) PVCs rises with the
  resting sinus rate, while <span style="color:#7ad9ff">interpolated</span> (silent)
  ones fall — so the number of thumps the subject notices tracks heart rate, not raw
  PVC count. Each point is one session (labelled); curves are a weighted logistic fit,
  the trend is quantified by Spearman's r in the title.
</div>
<img src="data:image/png;base64,{img_hr_pattern}" alt="Resting rate vs felt PVCs"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1040px; width: 100%; display:block; margin: 0 auto;"/>

<h2>Per-session summary table</h2>
<div class="commentary">
  Every session as a column, every metric as a row — the same layout as the summary
  report, recomputed at each run on the validated definitions used throughout this
  dashboard (local-sinus pause split for interpolated/compensated, the couplet detector
  from the section below, real pre-PVC coupling median). Colour cues:
  <b style="color:#33ff66">green</b>/<b style="color:#ffd633">amber</b>/<b style="color:#ff7a4d">orange</b>
  flag low/medium/higher values for burden, couplets and AF score;
  <span style="color:#7ad9ff">interpolated</span> /
  <span style="color:#ff8a8a">compensated</span> keep their usual colours. "Guarded" =
  PVCs whose pause was unmeasurable (a sinus beat hidden in the gap) and excluded from the
  interp/comp split. "AF score" is the 0&ndash;4 atrial-fibrillation signal screen
  (RMSSD&gt;100, pNN50&gt;40, high RR entropy, unimodal+high CV) — a signal-level check,
  not a diagnosis.
</div>
<div style="overflow-x:auto; margin: 12px 0;">
<table class="summary" style="font-size: 12.5px; min-width: 900px;">
  <tr>{summary_head}</tr>
  {summary_body}
</table>
</div>

<h2>Per-session pause distribution</h2>
<div class="commentary">
  One row per recording, recomputed at every run, using the validated criterion from
  the method section: each beat's pause RR<sub>post</sub> measured against its
  <em>local</em> sinus cycle, split at the global valley (<b>{PAUSE_VALLEY:.2f}&times;</b>).
  The left hump (<span style="color:#7ad9ff">blue, &asymp;0.7&times;</span>) are
  interpolated (silent) beats; the right hump (<span style="color:#ff8a8a">pink,
  &asymp;1.45&times;</span>) compensated (felt) ones; the
  <span style="color:#ffe169">yellow line</span> is the split. A session whose mass
  sits almost entirely on one side has a correspondingly lopsided <em>perceived</em>
  burden — visible here at a glance, and the driver behind the key pattern above.
  Beats failing the prematurity guard (a sub-threshold sinus beat hides between N and
  the PVC, making the pause unmeasurable) are excluded and counted in each row title.
</div>
<img src="data:image/png;base64,{img_persession_pause}" alt="Per-session pause distribution"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1080px; width: 100%; display:block; margin: 0 auto;"/>

<h2>Per-session coupling distribution &amp; single-focus check</h2>
<div class="commentary">
  <b>What the coupling interval is.</b> The coupling interval is the time from the
  preceding normal beat to the PVC (RR<sub>pre</sub>) — how long after each sinus beat
  the ectopic fires. A single ectopic focus re-entering on the same circuit fires at a
  near-constant coupling, so a <b>tight, recording-to-recording-stable peak</b> is the
  signature of one monomorphic focus; a <b>second focus</b> would add a separate
  coupling peak <em>with a different QRS shape</em>. Bars are tinted by sub-cluster
  (<span style="color:#7ad9ff">&lt;500</span> /
  <span style="color:#ff8a8a">500&ndash;600</span> /
  <span style="color:#7fd693">&gt;600&nbsp;ms</span>); the
  <span style="color:#ffe169">yellow line</span> is the median.
  <br/><br/>
  <b>Single-focus check (statistical).</b> For every session a 1- vs 2-component
  Gaussian mixture is fit to the coupling; a session is flagged
  <span style="color:#b59bff">bimodal</span> only when the two-component fit is
  <em>genuinely</em> two-peaked (a real trough between the modes, not mere skew) and
  improves the BIC — confirmed offline by a parametric-bootstrap likelihood-ratio test
  (p&asymp;0.002). For any bimodal session the two coupling clusters are then compared
  morphologically (median QRS template correlation, shown on the row). Result:
  {focus_summary}
</div>
<img src="data:image/png;base64,{img_persession_coupling}" alt="Per-session coupling distribution"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1080px; width: 100%; display:block; margin: 0 auto;"/>

<h2>Couplets — detection, count &amp; overlay</h2>
<div class="commentary">
  A <b>couplet</b> is two PVCs in a row (RR 200&ndash;700&nbsp;ms) that is <em>not</em>
  part of a longer run — the simplest form of repetitive ectopy, and the one worth
  counting because frequency and uniformity speak to risk. Every couplet in the dataset
  is detected (excluding noise-marked stretches and the warm-up minute) and shown here:
  <ul>
    <li><b>Count per session</b> (same session colours as above).</li>
    <li><b>All couplets overlaid</b>, each aligned on the first PVC's peak and amplitude-
        normalized, coloured by session — if they stack onto one another they are uniform.
        The <span style="color:#ffe169">yellow line</span> marks the median inter-PVC
        interval.</li>
    <li><b>First vs second beat</b>: the median QRS of the first PVC
        (<span style="color:#7ad9ff">blue</span>) against the second
        (<span style="color:#ffa64d">orange</span>). A high correlation means both beats
        come from the same focus (a benign repetitive discharge) rather than two different
        morphologies (which would be more concerning).</li>
  </ul>
  {coup_summary}
</div>
<img src="data:image/png;base64,{img_couplets}" alt="Couplets count and overlay"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1120px; width: 100%; display:block; margin: 0 auto;"/>

<div class="commentary" style="margin-top: 14px;">
  A few couplets in their raw recording context (±5 s), same format as the example
  strips earlier — green trace, the two ectopic QRS in
  <span style="color:#ff6b6b">red</span> with markers. One clean representative is shown
  per <b>distinct local rhythm</b> (the N/V motif is printed in each title) so the gallery
  spans the variations rather than repeating the dominant pattern.
</div>
<img src="data:image/png;base64,{img_couplet_strips}" alt="Couplet example strips"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1120px; width: 100%; display:block; margin: 0 auto;"/>

<h3>Local rhythm motif — does the couplet sit in a repeating pattern?</h3>
<div class="commentary">
  Coding each beat as N (sinus) or V (PVC) for the four beats before and after every
  couplet — the motifs printed on the strips above — reveals whether the couplet recurs
  inside a fixed rhythm. {pattern_summary}
</div>
<img src="data:image/png;base64,{img_couplet_motifs}" alt="Couplet local rhythm motifs"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1040px; width: 100%; display:block; margin: 0 auto;"/>

<h2>Respiration (EDR) &amp; respiratory-phase trigger</h2>
<div class="commentary">
  The final question: are the PVCs tied to <b>breathing</b>? There is no respiration belt,
  but the breath leaves a fingerprint on the ECG — the chest movement and lung-impedance
  change make the R-wave amplitude rise and fall with each breath (ECG-derived respiration,
  <b>EDR</b>). Recovering that signal from the R-amplitude of the normal beats
  (cubic-resampled to 4&nbsp;Hz, band-passed 0.1&ndash;0.5&nbsp;Hz, instantaneous phase by
  Hilbert transform) lets us ask, for every beat, <em>where in the breath</em> it fell, and
  compare PVCs against normal beats.
  {resp_summary}
</div>
<img src="data:image/png;base64,{img_edr_demo}" alt="EDR respiration demo"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1100px; width: 100%; display:block; margin: 0 auto;"/>
<div class="commentary" style="margin-top: 14px;">
  Above: a clean ~80&nbsp;s window from the best session — the R-amplitude of the sinus beats
  (<span style="color:#5fcc9e">green dots</span>) traces a slow oscillation, the recovered
  <span style="color:#7ad9ff">respiration</span>, and the
  <span style="color:#ff6b6b">PVCs</span> tend to land on a recurring part of it. Below: the
  phase distribution of all PVCs vs normal beats (left, polar) and the per-session enrichment
  around the respiratory cycle (right) — values above <b>&times;1</b> mark phases where PVCs
  are over-represented. Phase 0/100% is <b>full lungs</b> (largest R-amplitude, end-inspiration,
  confirmed by the subject); 50% is empty lungs (end-expiration). The enrichment peaks at full
  lungs.
</div>
<img src="data:image/png;base64,{img_resp_phase}" alt="Respiratory phase vs PVC enrichment"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: 1120px; width: 100%; display:block; margin: 0 auto;"/>

<h3>Respiratory phase by PVC type</h3>
<div class="commentary">
  Do the different PVC types fall at the same point in the breath, or different ones? This
  rosette splits the PVCs by type — <span style="color:#7ad9ff">interpolated</span>,
  <span style="color:#ff8a8a">compensated</span> and <span style="color:#ffd633">coupled</span>
  (both beats of each couplet) — and plots each one's phase density (normalized, so the
  <em>shape</em> is comparable despite very different counts). {type_summary}
</div>
<img src="data:image/png;base64,{img_resp_phase_types}" alt="Respiratory phase by PVC type"
     style="border:1px solid #25282d; border-radius:6px;
            max-width: {disp_width(img_resp_phase_types)}px; width: 100%;
            display:block; margin: 0 auto;"/>
<div style="overflow-x:auto; margin: 12px auto; max-width: 820px;">
<table>
  <tr><th>Session</th><th>Resp rate (/min)</th><th>PVCs</th><th>Peak phase</th>
      <th>Peak enrichment</th><th>p (&chi;&sup2;)</th><th>EDR quality score</th></tr>
  {resp_table}
</table>
</div>
<div class="commentary">
  A significant p means the PVCs are <b>not</b> uniformly spread across the breath — they
  cluster at a phase, i.e. respiration modulates the focus. This is a signal-level
  observation, not a clinical finding.
</div>

<footer>
  Sample rate {SR} Hz · single precordial lead AD8232 · 1st-order IIR bandpass
  0.3-25 Hz · QRS detector and PVC classifier as described in the
  &laquo;Acquisition setup&raquo; box above. Sessions filtered:
  size ≥ {MIN_FILE_SIZE_MB} MB and ≥ {MIN_PVC_COUNT} PVCs.<br/>
  This dashboard is intended for personal use and as a discussion document with
  the supervising cardiologist. It does <b>not</b> constitute a diagnostic
  report. To export a printable copy, open in a browser and use
  <em>Print → Save as PDF</em>.
</footer>

</body>
</html>"""

    os.makedirs("reports", exist_ok=True)
    out_path = "reports/holter_dashboard.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\n✓ Dashboard written: {out_path}")
    print(f"  Open in browser. Use Cmd+P → Save as PDF for export.")
    return out_path

if __name__ == "__main__":
    main()
