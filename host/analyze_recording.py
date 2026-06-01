"""
Detector v3: dual-threshold + REBOUND (iperpolarizzazione post-QRS).
State machine: IDLE -> WIDTH -> DETECT -> POST(200ms misuro trough) -> IDLE.
PVC se rebound_ratio = |trough|/peak >= 0.40, OPPURE width >= 95ms.
"""
import math
import statistics
import sys
import numpy as np
import matplotlib.pyplot as plt

PATH = sys.argv[1] if len(sys.argv) > 1 else "ecg_30s.csv"
SAMPLE_HZ = 250

HP_FC = 0.3
LP_FC = 25.0
REFRACTORY_S = 0.30
WIDTH_THR = 0.10
DETECT_THR_FLOOR = 0.30
DETECT_THR_RATIO = 0.45
POST_PEAK_MS = 200
REBOUND_RATIO_PVC = 0.40
PVC_WIDTH_MS = 95.0

class HPFilter:
    def __init__(self, fc, fs):
        RC = 1.0 / (2 * math.pi * fc); dt = 1.0 / fs
        self.alpha = RC / (RC + dt); self.y = 0.0; self.x = 0.0
    def __call__(self, x):
        y = self.alpha * (self.y + x - self.x); self.y = y; self.x = x; return y

class LPFilter:
    def __init__(self, fc, fs):
        RC = 1.0 / (2 * math.pi * fc); dt = 1.0 / fs
        self.beta = dt / (RC + dt); self.y = 0.0
    def __call__(self, x):
        y = self.beta * x + (1 - self.beta) * self.y; self.y = y; return y

# load
t_us, volt = [], []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split(",")
        if len(parts) < 3: continue
        try:
            t_us.append(int(parts[0])); volt.append(float(parts[2]))
        except ValueError:
            continue

t = np.array(t_us) / 1e6
v_raw = np.array(volt)
print(f"loaded {len(v_raw)} samples, duration {t[-1]-t[0]:.2f}s, rate {len(v_raw)/(t[-1]-t[0]):.2f} Hz")

hp = HPFilter(HP_FC, SAMPLE_HZ); lp = LPFilter(LP_FC, SAMPLE_HZ)
v_f = np.array([lp(hp(x)) for x in v_raw])

REFRACTORY_SAMPLES = int(REFRACTORY_S * SAMPLE_HZ)
SETTLING_SAMPLES = int(1.5 * SAMPLE_HZ)
POST_PEAK_SAMPLES = int(POST_PEAK_MS * SAMPLE_HZ / 1000)

STATE_IDLE, STATE_WIDTH, STATE_DETECT, STATE_POST = 0, 1, 2, 3

peaks = []  # (sample_idx, amp, width_ms, trough, rebound_ratio, classification)
peak_amplitudes = []
fsm = STATE_IDLE
width_start = None
peak_amp = 0.0
peak_n = 0
last_peak_sample = None
post_counter = 0
post_trough = 0.0

def detect_threshold():
    if len(peak_amplitudes) >= 3:
        return max(DETECT_THR_FLOOR, statistics.median(peak_amplitudes) * DETECT_THR_RATIO)
    return DETECT_THR_FLOOR

def finalize_pending(peak_n, peak_amp, width_ms, trough):
    rebound = abs(trough) if trough < 0 else 0.0
    ratio = rebound / peak_amp if peak_amp > 0 else 0.0
    cls = "normal"
    if ratio >= REBOUND_RATIO_PVC or width_ms >= PVC_WIDTH_MS:
        cls = "pvc"
    return cls, ratio

for n, v in enumerate(v_f):
    if n < SETTLING_SAMPLES:
        continue
    dthr = detect_threshold()

    if fsm == STATE_IDLE:
        if v > WIDTH_THR:
            fsm = STATE_WIDTH
            width_start = n
            peak_amp = v; peak_n = n

    elif fsm == STATE_WIDTH:
        if v > peak_amp: peak_amp = v; peak_n = n
        if v > dthr:
            fsm = STATE_DETECT
        elif v < WIDTH_THR * 0.8:
            fsm = STATE_IDLE

    elif fsm == STATE_DETECT:
        if v > peak_amp: peak_amp = v; peak_n = n
        if v < WIDTH_THR:
            # fine del corpo del QRS; ora monitoro il rebound
            pending_width_ms = (n - width_start) * 1000.0 / SAMPLE_HZ
            pending_start = width_start
            pending_peak_n = peak_n
            pending_peak_amp = peak_amp
            post_counter = 0
            post_trough = v
            fsm = STATE_POST

    elif fsm == STATE_POST:
        if v < post_trough:
            post_trough = v
        post_counter += 1
        # finalizza dopo POST_PEAK_SAMPLES o se un nuovo QRS sta partendo (oltre refractory)
        new_qrs_starting = (v > WIDTH_THR and post_counter > REFRACTORY_SAMPLES)
        if post_counter >= POST_PEAK_SAMPLES or new_qrs_starting:
            # accept se fuori refractory dal precedente
            if last_peak_sample is None or (pending_peak_n - last_peak_sample) > REFRACTORY_SAMPLES:
                cls, ratio = finalize_pending(pending_peak_n, pending_peak_amp,
                                              pending_width_ms, post_trough)
                peak_amplitudes.append(pending_peak_amp)
                peaks.append((pending_peak_n, pending_peak_amp, pending_width_ms,
                              post_trough, ratio, cls))
                last_peak_sample = pending_peak_n
            # se un nuovo QRS e' gia' iniziato, transita direttamente
            if new_qrs_starting:
                fsm = STATE_WIDTH
                width_start = n
                peak_amp = v; peak_n = n
            else:
                fsm = STATE_IDLE

n_total = len(peaks)
n_pvc = sum(1 for *_, c in peaks if c == "pvc")
n_normal = n_total - n_pvc

widths_normal = [w for _,_,w,_,_,c in peaks if c == "normal"]
widths_pvc    = [w for _,_,w,_,_,c in peaks if c == "pvc"]
ratios_normal = [r for _,_,_,_,r,c in peaks if c == "normal"]
ratios_pvc    = [r for _,_,_,_,r,c in peaks if c == "pvc"]
rrs = [(peaks[i][0]-peaks[i-1][0])/SAMPLE_HZ for i in range(1, len(peaks))]

print(f"\n=== DETECTION (rebound + width) ===")
print(f"battiti totali: {n_total}  (normali: {n_normal}, PVC: {n_pvc})")
print(f"BPM totale:   {n_total/t[-1]*60:.1f}")
print(f"BPM sinusale: {n_normal/t[-1]*60:.1f}")
print(f"PVC rate:     {n_pvc/t[-1]*60:.1f}/min")
print(f"PVC burden:   {100*n_pvc/n_total if n_total else 0:.1f}%")
if widths_normal:
    print(f"\nwidth normali: med {statistics.median(widths_normal):.0f}ms, range {min(widths_normal):.0f}-{max(widths_normal):.0f}")
if widths_pvc:
    print(f"width PVC:     med {statistics.median(widths_pvc):.0f}ms, range {min(widths_pvc):.0f}-{max(widths_pvc):.0f}")
if ratios_normal:
    print(f"\nrebound ratio normali: med {statistics.median(ratios_normal):.2f}, range {min(ratios_normal):.2f}-{max(ratios_normal):.2f}")
if ratios_pvc:
    print(f"rebound ratio PVC:     med {statistics.median(ratios_pvc):.2f}, range {min(ratios_pvc):.2f}-{max(ratios_pvc):.2f}")
if rrs:
    print(f"\nRR mediana: {statistics.median(rrs)*1000:.0f}ms ({60/statistics.median(rrs):.0f} BPM)")

print(f"\n=== ALL PEAKS ===")
print(f"{'#':3s} {'t(s)':>6s} {'amp':>6s} {'width':>6s} {'trough':>7s} {'reb':>5s} {'class':>7s}")
for i, (p_idx, p_amp, w_ms, tr, r, cls) in enumerate(peaks):
    print(f"{i+1:3d} {t[p_idx]:6.2f} {p_amp:6.2f} {w_ms:6.0f} {tr:7.3f} {r:5.2f} {cls:>7s}")

# plot
fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
fig.patch.set_facecolor('#1e1e1e')

ax0 = axes[0]; ax0.set_facecolor('#0d0d0d')
ax0.plot(t, v_raw, linewidth=0.7, color='#888888')
ax0.set_ylabel("Raw (V)", color='#aaaaaa')
ax0.set_title(f"Raw ECG — {PATH}", color='#ffffff')
ax0.grid(True, alpha=0.2, color='#444444'); ax0.tick_params(colors='#aaaaaa')

ax1 = axes[1]; ax1.set_facecolor('#0d0d0d')
ax1.plot(t, v_f, linewidth=0.8, color='#2ecc71')

for p_idx, p_amp, w_ms, tr, r, cls in peaks:
    pt = t[p_idx]
    if cls == "pvc":
        ax1.scatter(pt, p_amp + 0.12, s=200, marker='v', color='#e74c3c',
                    edgecolors='white', zorder=5)
        ax1.text(pt, p_amp + 0.30, f"r={r:.2f}\n{w_ms:.0f}ms",
                 color='#e74c3c', fontsize=8, ha='center', fontweight='bold')
    else:
        ax1.scatter(pt, p_amp + 0.10, s=70, marker='v', color='#2ecc71',
                    edgecolors='white', zorder=5)

ax1.axhline(WIDTH_THR, color='#3498db', linestyle=':', linewidth=0.7, alpha=0.6,
            label=f'width thr: {WIDTH_THR}V')
ax1.axhline(detect_threshold(), color='#f39c12', linestyle='--', linewidth=0.8, alpha=0.7,
            label=f'detect thr: {detect_threshold():.2f}V')
ax1.legend(loc='upper right', facecolor='#222222', edgecolor='#444444', labelcolor='#ffffff')
ax1.set_xlabel("Time (s)", color='#aaaaaa')
ax1.set_ylabel("Filtered (V)", color='#aaaaaa')
ax1.set_title(f"PVC = rebound>=0.40 OR width>=95ms — {n_normal} normali, {n_pvc} PVC",
              color='#ffffff')
ax1.grid(True, alpha=0.2, color='#444444'); ax1.tick_params(colors='#aaaaaa')
for ax in axes:
    for sp in ax.spines.values(): sp.set_color('#444444')

plt.tight_layout()
out = PATH.replace(".csv", "_rebound.png")
plt.savefig(out, dpi=120, facecolor='#1e1e1e')
print(f"\nplot saved: {out}")
