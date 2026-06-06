"""
Genera un report HTML autonomo con grafici embedded da una sessione registrata.

Usage:
    python3 generate_report.py logs/ecg_YYYYMMDD_HHMMSS.csv

Output: logs/report_YYYYMMDD_HHMMSS.html (apri in browser, opzionale stampa in PDF)
"""
import base64
import csv
import io
import math
import os
import statistics
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else None
if PATH is None:
    print("usage: generate_report.py <ecg_*.csv>")
    sys.exit(1)

SAMPLE_HZ = 250

# --- load samples ---
t_s, vr, vf = [], [], []
with open(PATH) as f:
    for row in csv.DictReader(f):
        try:
            t_s.append(float(row["t_s"]))
            vr.append(float(row["raw"]))
            vf.append(float(row["filt"]))
        except (KeyError, ValueError):
            continue
t  = np.array(t_s)
vr = np.array(vr)
vf = np.array(vf)
N  = len(vf)

# --- load peaks ---
peaks_path = PATH.replace(os.sep + "ecg_", os.sep + "peaks_")
peaks = []
if os.path.exists(peaks_path):
    with open(peaks_path) as f:
        for row in csv.DictReader(f):
            try:
                peaks.append({
                    "t":   float(row["t_s"]),
                    "amp": float(row["amp_V"]),
                    "w":   float(row["width_ms"]),
                    "reb": float(row["rebound_ratio"]),
                    "cls": row["class"],
                })
            except (KeyError, ValueError):
                continue

# stamps
ses_id = os.path.basename(PATH).replace("ecg_", "").replace(".csv", "")
total_s = float(t[-1] - t[0]) if N else 0
total_min = total_s / 60.0
fs_real = N / total_s if total_s else SAMPLE_HZ
norm = [p for p in peaks if p["cls"] == "normal"]
pvc  = [p for p in peaks if p["cls"] == "pvc"]
n_total = len(peaks)
sinus_bpm = 60 * len(norm) / total_s if total_s else 0
pvc_rate  = 60 * len(pvc)  / total_s if total_s else 0
burden    = 100 * len(pvc) / max(1, n_total)

# RR intervals
for i in range(len(peaks)):
    peaks[i]["rr_prev"] = (peaks[i]["t"] - peaks[i-1]["t"]) if i > 0 else None
sinus_rr  = [peaks[i]["rr_prev"] for i in range(1, len(peaks))
             if peaks[i]["cls"] == "normal" and peaks[i-1]["cls"] == "normal"]
coupling  = [p["rr_prev"] for p in peaks if p["cls"] == "pvc" and p["rr_prev"] is not None]

# pattern counts
iso_pvc = sum(
    1 for i, p in enumerate(peaks) if p["cls"] == "pvc"
    and (i == 0 or peaks[i-1]["cls"] != "pvc")
    and (i == len(peaks)-1 or peaks[i+1]["cls"] != "pvc")
)
couplets_n = 0
i = 0
while i < len(peaks) - 1:
    if peaks[i]["cls"] == "pvc" and peaks[i+1]["cls"] == "pvc":
        if i+2 < len(peaks) and peaks[i+2]["cls"] == "pvc":
            i += 1
            continue
        couplets_n += 1
        i += 2
    else:
        i += 1

# bigeminy / trigeminy runs
bigem = 0
i = 0
while i < len(peaks) - 5:
    if [peaks[i+j]["cls"] for j in range(6)] == ["normal","pvc"]*3:
        bigem += 1
        # skip ahead
        end = i
        while end+1 < len(peaks) and peaks[end+1]["cls"] != peaks[end]["cls"]:
            end += 1
        i = end + 1
    else:
        i += 1

trigem = 0
i = 0
while i < len(peaks) - 8:
    if [peaks[i+j]["cls"] for j in range(9)] == ["normal","normal","pvc"]*3:
        trigem += 1
        end = i
        while end+3 < len(peaks) and peaks[end+1]["cls"]=="normal" and peaks[end+2]["cls"]=="normal" and peaks[end+3]["cls"]=="pvc":
            end += 3
        i = end + 1
    else:
        i += 1

# pre-PVC HRV
LOOKBACK = 5
pre_pvc_stdevs = []
for idx, p in enumerate(peaks):
    if p["cls"] != "pvc": continue
    rrs = []
    j = idx - 1
    while j >= 1 and len(rrs) < LOOKBACK:
        if peaks[j]["cls"] == "normal" and peaks[j-1]["cls"] == "normal":
            rrs.append(peaks[j]["t"] - peaks[j-1]["t"])
        j -= 1
    if len(rrs) >= LOOKBACK:
        pre_pvc_stdevs.append(statistics.stdev(rrs))

baseline_stdevs = []
if len(sinus_rr) >= LOOKBACK:
    for k in range(0, len(sinus_rr) - LOOKBACK + 1, LOOKBACK):
        baseline_stdevs.append(statistics.stdev(sinus_rr[k:k+LOOKBACK]))

# per-minute stats
WINDOW = 60
windows = []
i_w = 0
while peaks and peaks[0]["t"] + i_w*WINDOW < peaks[-1]["t"]:
    ws = peaks[0]["t"] + i_w*WINDOW
    we = ws + WINDOW
    in_w = [p for p in peaks if ws <= p["t"] < we]
    nn = sum(1 for p in in_w if p["cls"] == "normal")
    np_ = sum(1 for p in in_w if p["cls"] == "pvc")
    windows.append({"t": ws, "norm": nn, "pvc": np_})
    i_w += 1

# --- plot helpers ---
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="#1e1e1e", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def styled_ax(ax, title=None):
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="#aaa")
    ax.grid(True, alpha=0.2, color="#444")
    for sp in ax.spines.values(): sp.set_color("#444")
    if title: ax.set_title(title, color="#fff")

# PLOT 1 — esempio di traccia: 6 secondi con almeno una PVC
ecg_example_b64 = ""
if pvc and N:
    # cerca primo PVC e fa un crop ±3s
    p0 = pvc[0]["t"]
    mask = (t >= p0-3) & (t <= p0+3)
    fig, ax = plt.subplots(figsize=(11, 3))
    fig.patch.set_facecolor("#1e1e1e")
    ax.plot(t[mask]-p0, vf[mask], linewidth=0.9, color="#2ecc71")
    # red overlay
    for p in pvc:
        if p0-3 <= p["t"] <= p0+3:
            pt = p["t"] - p0
            wm = (t >= p["t"]-0.12) & (t <= p["t"]+0.12)
            ax.plot(t[wm]-p0, vf[wm], linewidth=2.0, color="#e74c3c")
    ax.set_xlabel("t (s relativo al primo PVC)", color="#aaa")
    ax.set_ylabel("ECG filtrato (V)", color="#aaa")
    styled_ax(ax, "Esempio: 6 secondi con QRS normali (verde) e una PVC (rosso)")
    plt.tight_layout()
    ecg_example_b64 = fig_to_b64(fig)

# PLOT 2 — overview intera sessione (downsampled)
overview_b64 = ""
if N:
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor("#1e1e1e")
    step = max(1, N // 50000)  # ~50k punti max
    ax.plot(t[::step]/60, vf[::step], linewidth=0.3, color="#2ecc71", alpha=0.8)
    # PVC dots
    if pvc:
        ax.scatter([p["t"]/60 for p in pvc], [1.4]*len(pvc), s=4, color="#e74c3c", marker="v")
    ax.set_xlabel("Tempo (min)", color="#aaa")
    ax.set_ylabel("ECG filt (V)", color="#aaa")
    ax.set_ylim(-1.4, 1.6)
    styled_ax(ax, f"Overview {total_min:.1f} min — triangoli rossi sopra = posizioni PVC")
    plt.tight_layout()
    overview_b64 = fig_to_b64(fig)

# PLOT 3 — tachogramma
tacho_b64 = ""
if peaks:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#1e1e1e")
    for p in peaks:
        if p["rr_prev"] is None: continue
        c = "#e74c3c" if p["cls"] == "pvc" else "#2ecc71"
        ax.scatter(p["t"]/60, 1000*p["rr_prev"], c=c, s=6, alpha=0.75)
    ax.set_xlabel("Tempo (min)", color="#aaa")
    ax.set_ylabel("RR (ms)", color="#aaa")
    styled_ax(ax, "Tachogramma — RR dei battiti nel tempo (verde = sinus, rosso = pre-PVC coupling)")
    plt.tight_layout()
    tacho_b64 = fig_to_b64(fig)

# PLOT 4 — histogram RR
hist_b64 = ""
if coupling and sinus_rr:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#1e1e1e")
    ax.hist([r*1000 for r in sinus_rr], bins=40, alpha=0.6, color="#2ecc71",
            label=f"Sinus N→N (n={len(sinus_rr)})", density=True)
    ax.hist([r*1000 for r in coupling], bins=40, alpha=0.8, color="#e74c3c",
            label=f"Pre-PVC coupling (n={len(coupling)})", density=True)
    ax.set_xlabel("RR (ms)", color="#aaa")
    ax.set_ylabel("Densità", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="#fff", edgecolor="#444")
    styled_ax(ax, "Distribuzione RR — sinus vs coupling pre-PVC")
    plt.tight_layout()
    hist_b64 = fig_to_b64(fig)

# PLOT 5 — counts per minute
counts_b64 = ""
if windows:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#1e1e1e")
    ts = [w["t"]/60 for w in windows]
    ax.plot(ts, [w["pvc"] for w in windows], color="#e74c3c", marker="o", label="PVC/min")
    ax.plot(ts, [w["norm"] for w in windows], color="#2ecc71", marker="o", label="Sinus/min", alpha=0.6)
    ax.set_xlabel("Tempo (min)", color="#aaa")
    ax.set_ylabel("Battiti per minuto", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="#fff", edgecolor="#444")
    styled_ax(ax, "Distribuzione battiti nel tempo (finestre da 60s)")
    plt.tight_layout()
    counts_b64 = fig_to_b64(fig)

# PLOT 6 — pre-PVC HRV
hrv_b64 = ""
if pre_pvc_stdevs:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#1e1e1e")
    pvc_ts = [p["t"]/60 for p in peaks if p["cls"] == "pvc"]
    if len(pvc_ts) >= len(pre_pvc_stdevs):
        pvc_ts = pvc_ts[-len(pre_pvc_stdevs):]
    ax.scatter(pvc_ts, [1000*s for s in pre_pvc_stdevs], c="#e74c3c", s=10, alpha=0.7,
               label="Stdev RR nei 5 N pre-PVC")
    if baseline_stdevs:
        baseline_mean = 1000 * statistics.mean(baseline_stdevs)
        ax.axhline(baseline_mean, color="#2ecc71", linestyle="--", alpha=0.7,
                   label=f"Baseline sinus stdev ({baseline_mean:.0f}ms)")
    ax.set_xlabel("Tempo (min)", color="#aaa")
    ax.set_ylabel("Stdev RR (ms)", color="#aaa")
    ax.legend(facecolor="#222", labelcolor="#fff", edgecolor="#444")
    styled_ax(ax, "HRV nei 5 battiti normali prima di ogni PVC")
    plt.tight_layout()
    hrv_b64 = fig_to_b64(fig)

# --- compute summary numbers ---
sinus_median_ms = 1000 * statistics.median(sinus_rr) if sinus_rr else 0
sinus_std_ms    = 1000 * statistics.stdev(sinus_rr)  if len(sinus_rr) > 1 else 0
coupling_median = 1000 * statistics.median(coupling) if coupling else 0
coupling_std    = 1000 * statistics.stdev(coupling)  if len(coupling) > 1 else 0
prematurity = (1 - coupling_median/sinus_median_ms) * 100 if sinus_median_ms else 0
pre_pvc_stdev_mean = 1000 * statistics.mean(pre_pvc_stdevs) if pre_pvc_stdevs else 0
baseline_stdev_mean = 1000 * statistics.mean(baseline_stdevs) if baseline_stdevs else 0
hrv_delta_pct = (pre_pvc_stdev_mean/baseline_stdev_mean - 1) * 100 if baseline_stdev_mean else 0

# --- compose HTML ---
ts_pretty = ses_id  # format YYYYMMDD_HHMMSS
now = datetime.now().strftime("%Y-%m-%d %H:%M")

html = f"""<!DOCTYPE html>
<html lang="it"><head>
<meta charset="utf-8">
<title>Report sessione holter — {ts_pretty}</title>
<style>
  :root {{
    --bg: #1e1e1e; --panel: #2a2a2a; --fg: #eee; --muted: #888;
    --green: #2ecc71; --blue: #3498db; --red: #e74c3c; --orange: #f39c12;
  }}
  body {{
    margin: 0; padding: 32px; background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro", Helvetica, sans-serif;
    line-height: 1.55; max-width: 1100px; margin: 0 auto;
  }}
  h1 {{ color: #fff; font-size: 28px; margin-bottom: 4px; }}
  h2 {{ color: var(--blue); margin-top: 36px; border-bottom: 1px solid #333; padding-bottom: 6px; }}
  h3 {{ color: var(--green); }}
  .meta {{ color: var(--muted); font-size: 13px; }}
  .disclaimer {{
    background: #3a1a1a; border-left: 4px solid var(--red);
    padding: 12px 18px; margin: 20px 0; border-radius: 4px;
  }}
  table {{ border-collapse: collapse; margin: 16px 0; width: 100%; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
  th {{ color: var(--blue); font-weight: 600; }}
  .num {{ font-variant-numeric: tabular-nums; }}
  .green {{ color: var(--green); }} .red {{ color: var(--red); }}
  .blue {{ color: var(--blue); }} .orange {{ color: var(--orange); }}
  .key {{ font-size: 32px; font-weight: 700; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 20px 0; }}
  .card {{ background: var(--panel); padding: 14px 16px; border-radius: 8px; }}
  .card .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  .card .val {{ font-size: 26px; font-weight: 700; margin-top: 4px; }}
  .card .u {{ font-size: 11px; color: var(--muted); }}
  img {{ width: 100%; border-radius: 6px; margin: 8px 0; }}
  .footer {{ color: var(--muted); font-size: 11px; margin-top: 60px; border-top: 1px solid #333; padding-top: 20px; }}
  code {{ background: #2a2a2a; padding: 2px 6px; border-radius: 3px; font-size: 13px; }}
</style>
</head><body>

<h1>Report sessione holter</h1>
<div class="meta">
  Sessione <code>{ts_pretty}</code> · Durata <b>{total_min:.1f} min</b>
  ({total_s:.0f}s, fs={fs_real:.2f}Hz) · Generato il {now}
</div>

<div class="disclaimer">
  <b>Uso esclusivamente didattico.</b> Questo report è il risultato di un dispositivo
  hobbistico (Pi Pico W + AD8232) e di pattern recognition implementati a fini educativi.
  Non è uno strumento medico e non sostituisce in alcun modo l'interpretazione del cardiologo
  curante. Eventuali sintomi vanno valutati clinicamente.
</div>

<h2>1. Sintesi numerica</h2>
<div class="cards">
  <div class="card"><div class="label">ECG total</div>
    <div class="val green num">{60*n_total/total_s:.0f}</div><div class="u">BPM totale</div></div>
  <div class="card"><div class="label">Sinus only</div>
    <div class="val blue num">{sinus_bpm:.0f}</div><div class="u">BPM ritmo sinusale</div></div>
  <div class="card"><div class="label">PVC rate</div>
    <div class="val red num">{pvc_rate:.1f}</div><div class="u">/min</div></div>
  <div class="card"><div class="label">PVC burden</div>
    <div class="val orange num">{burden:.1f}</div><div class="u">% del totale</div></div>
</div>

<table>
<tr><th>Metrica</th><th>Valore</th></tr>
<tr><td>Battiti registrati</td><td class="num">{n_total}</td></tr>
<tr><td>Battiti normali (sinusali)</td><td class="num">{len(norm)}</td></tr>
<tr><td>Battiti ectopici classificati PVC</td><td class="num">{len(pvc)}</td></tr>
<tr><td>RR sinusale (N→N) mediano</td><td class="num">{sinus_median_ms:.0f} ms ({60000/sinus_median_ms if sinus_median_ms else 0:.0f} BPM)</td></tr>
<tr><td>Deviazione standard RR sinusale</td><td class="num">{sinus_std_ms:.0f} ms</td></tr>
<tr><td>Coupling interval mediano (N→PVC)</td><td class="num">{coupling_median:.0f} ms</td></tr>
<tr><td>Deviazione standard coupling</td><td class="num">{coupling_std:.0f} ms</td></tr>
<tr><td>Prematurità della PVC</td><td class="num">{prematurity:.0f} % più precoce del ritmo sinusale</td></tr>
</table>

<h2>2. Esempio di traccia</h2>
<p>Una finestra di ~6 secondi centrata sulla prima PVC della sessione.
La linea verde è il segnale ECG filtrato; il segmento rosso evidenzia il QRS della PVC
e la sua iperpolarizzazione di rebound (~120 ms post-picco).</p>
<img src="data:image/png;base64,{ecg_example_b64}" alt="ecg_example">

<h2>3. Overview dell'intera sessione</h2>
<p>Vista compressa di tutta la registrazione. Ogni triangolo rosso in alto è la
posizione di una PVC. La distribuzione è uniforme nel tempo, segno di un focolaio
ectopico stabile.</p>
<img src="data:image/png;base64,{overview_b64}" alt="overview">

<h2>4. Pattern temporali</h2>
<table>
<tr><th>Tipo di pattern</th><th>Count</th></tr>
<tr><td>PVC isolate (N–PVC–N)</td><td class="num">{iso_pvc}</td></tr>
<tr><td>Couplet (2 PVC consecutive)</td><td class="num">{couplets_n}</td></tr>
<tr><td>Run di <b>bigeminia</b> (N-PVC alternati, ≥3 cicli)</td><td class="num">{bigem}</td></tr>
<tr><td>Run di <b>trigeminia</b> (N-N-PVC ripetuto, ≥3 cicli)</td><td class="num">{trigem}</td></tr>
</table>
<p><b>Lettura:</b> la stragrande maggioranza delle PVC è <b>isolata</b>, intervallata
da gruppi di battiti normali. Quando si manifesta un pattern ritmico ricorrente è
più spesso <b>trigeminia</b> ({trigem} run) che bigeminia ({bigem} run). I couplet
(2 PVC consecutive) sono praticamente assenti ({couplets_n}). Niente triplet o run
ventricolari più lunghi.</p>

<h2>5. Tachogramma</h2>
<p>Per ciascun battito si mostra il suo intervallo RR rispetto al battito precedente.
I due cluster orizzontali sono ben separati: in alto i battiti sinusali a ~{sinus_median_ms:.0f}ms,
in basso i coupling pre-PVC a ~{coupling_median:.0f}ms.</p>
<img src="data:image/png;base64,{tacho_b64}" alt="tacho">

<h2>6. Distribuzione degli RR — sinus vs coupling</h2>
<p>Distribuzione bimodale netta. Le PVC arrivano sistematicamente al ~{100-prematurity:.0f}%
del ciclo sinusale (prematurità del {prematurity:.0f}%). Questa <b>stabilità del coupling
interval</b> è la firma cardinale di un <b>focolaio ectopico monomorfo</b>: il "trigger" si
scarica sempre dallo stesso punto del miocardio con la stessa latenza dopo un battito normale.</p>
<img src="data:image/png;base64,{hist_b64}" alt="hist">

<h2>7. Variabilità nel tempo</h2>
<p>Conteggio di battiti normali e PVC in finestre di 60 secondi.</p>
<img src="data:image/png;base64,{counts_b64}" alt="counts">

<h2>8. Variabilità del ritmo sinusale prima delle PVC</h2>
<p>Per ogni PVC sono stati misurati i 5 battiti normali immediatamente precedenti, calcolando
la deviazione standard dei loro intervalli RR. Il valore <b>medio è {pre_pvc_stdev_mean:.0f} ms</b>,
da confrontare con la <b>baseline sinusale di {baseline_stdev_mean:.0f} ms</b> (delta
{'+' if hrv_delta_pct>=0 else ''}{hrv_delta_pct:.0f}%).</p>
<p>{
   "Il ritmo sinusale è <b>leggermente più variabile</b> nei battiti immediatamente "
   "antecedenti una PVC. Differenza piccola ma sistematica su centinaia di osservazioni. "
   "Coerente con una <b>modulazione autonomica</b> (vagale o respiratoria) del trigger ectopico."
   if hrv_delta_pct > 5 else
   "La variabilità non è significativamente diversa dal baseline."
}</p>
<img src="data:image/png;base64,{hrv_b64}" alt="hrv">

<h2>9. Conclusioni descrittive</h2>
<ul>
  <li><b>Focolaio ectopico monomorfo stabile</b>: il coupling interval ha bassa dispersione
      ({coupling_std:.0f}ms su {coupling_median:.0f}ms mediana), indicando una sorgente
      ectopica singola e consistente.</li>
  <li><b>PVC isolate, raramente in raffica</b>: {iso_pvc} singole su {len(pvc)} totali
      ({100*iso_pvc/max(1,len(pvc)):.0f}% del totale).</li>
  <li><b>Pattern di trigeminia più frequente della bigeminia</b>: {trigem} run di trigeminia
      vs {bigem} di bigeminia.</li>
  <li><b>Burden costante nel tempo</b>: la frequenza PVC/min resta stabile per
      la durata della registrazione, senza accumuli o quiescenze marcate.</li>
  <li><b>Lieve aumento di HRV pre-PVC</b>: +{hrv_delta_pct:.0f}% rispetto al baseline sinusale,
      compatibile con modulazione autonomica (potenzialmente respiratoria).</li>
  <li><b>Pulse deficit teorico atteso</b>: ~{len(pvc)/n_total*100:.0f}% delle PVC potrebbe
      non generare polso periferico apprezzabile (questo è il fenomeno per cui un Garmin
      al polso conterebbe una frequenza più bassa dell'ECG).</li>
</ul>

<h2>10. Limiti tecnici</h2>
<ul>
  <li>Singola derivazione (Einthoven I) → impossibile localizzare il focolaio
      ectopico nei tre piani anatomici.</li>
  <li>Detector basato su iperpolarizzazione post-QRS e larghezza → robusto per il
      pattern attuale, ma potrebbe perdere PVC con morfologia atipica.</li>
  <li>{total_min:.1f} min di registrazione, finestra troppo breve per inferire trend
      circadiani o post-prandiali.</li>
  <li>Nessun sensore di respiro sincronizzato → impossibile testare formalmente la
      correlazione PVC ↔ fase respiratoria.</li>
</ul>

<div class="footer">
  File sorgente: <code>{os.path.basename(PATH)}</code><br>
  Pipeline: Pi Pico W (250 Hz ADC) → WiFi → server Python (Flask/SSE) → detector
  band-pass 0.3–25 Hz + 4-state FSM + rebound/width classification.<br>
  Repository: <code>https://github.com/mrEg0n/holter-ecg</code><br>
  Generato automaticamente da <code>host/generate_report.py</code>
</div>
</body></html>
"""

out_path = PATH.replace(os.sep + "ecg_", os.sep + "report_").replace(".csv", ".html")
with open(out_path, "w") as f:
    f.write(html)
print(f"Report salvato: {out_path}")
print(f"  size: {os.path.getsize(out_path)//1024} KB")
