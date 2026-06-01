"""
Live ECG dashboard with PVC (premature ventricular contraction) detection.

Pipeline:
  1. Launch streamer on Pico via mpremote subprocess; read stdout in a thread.
  2. Apply IIR band-pass filter (HP 0.3 Hz + LP 25 Hz) per sample.
  3. 4-state machine: IDLE -> WIDTH (above 0.10 V) -> DETECT (above ~0.30 V)
     -> POST (200 ms trough monitoring) -> IDLE.
  4. Classify each beat as PVC if rebound ratio (|trough| / peak) >= 0.40
     OR QRS width >= 95 ms. Otherwise normal.
  5. Render scrolling 30 s window, color-coded beat markers, live BPM
     readouts (ECG total / sinus only / PVC rate / burden %).

This is pattern recognition for didactic purposes, not a diagnostic tool.
"""
import subprocess
import threading
import collections
import statistics
import math
import os
import glob
import shutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def find_mpremote():
    """Try PATH first, fall back to ~/Library/Python/.../bin (pip --user on macOS)."""
    if shutil.which("mpremote"):
        return "mpremote"
    home = os.path.expanduser("~")
    candidates = glob.glob(os.path.join(home, "Library/Python/*/bin/mpremote"))
    candidates += glob.glob(os.path.join(home, ".local/bin/mpremote"))
    if candidates:
        return candidates[0]
    raise SystemExit("mpremote non trovato. Installa con: pip3 install --user mpremote")

def find_pico_device():
    """Auto-detect Pico USB serial device on macOS or Linux."""
    env = os.environ.get("PICO_DEVICE")
    if env:
        return env
    for pat in ("/dev/cu.usbmodem*", "/dev/ttyACM*"):
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    raise SystemExit("Nessun Pico rilevato. Collega via USB o imposta PICO_DEVICE=/dev/...")

MPREMOTE = find_mpremote()
DEVICE   = find_pico_device()
# streamer.py lives in ../pico/ relative to this script
STREAMER = os.environ.get(
    "STREAMER_PATH",
    os.path.normpath(os.path.join(HERE, "..", "pico", "streamer.py")),
)

SAMPLE_HZ = 250
WINDOW_S  = 30
BUF_SIZE  = SAMPLE_HZ * WINDOW_S

# Detector dual-threshold + REBOUND post-QRS
WIDTH_THR        = 0.10   # soglia bassa per misurare larghezza vera del QRS
DETECT_THR_FLOOR = 0.30   # soglia alta minima per confermare QRS (esclude T-wave)
DETECT_THR_RATIO = 0.45   # 45% della mediana ampiezze
POST_PEAK_MS     = 200    # ms da monitorare dopo ogni QRS per il rebound
REBOUND_RATIO_PVC = 0.40  # |trough|/peak >= 0.40 → PVC (iperpolarizzazione)
PVC_WIDTH_MS     = 95.0   # OPPURE width >= 95 ms → PVC (criterio secondario)

BPM_WINDOW_S  = 60        # finestra temporale per calcolo BPM (secondi)

# --- Filtri IIR semplici ---
class HPFilter:
    """High-pass 1st order. Toglie DC e deriva lenta. fc bassa per non distorcere QRS larghi."""
    def __init__(self, fc, fs):
        RC = 1.0 / (2 * math.pi * fc)
        dt = 1.0 / fs
        self.alpha = RC / (RC + dt)
        self.y = 0.0
        self.x = 0.0
    def __call__(self, x):
        y = self.alpha * (self.y + x - self.x)
        self.y = y
        self.x = x
        return y

class LPFilter:
    """Low-pass 1st order. Attenua >40Hz (mains, EMG)."""
    def __init__(self, fc, fs):
        RC = 1.0 / (2 * math.pi * fc)
        dt = 1.0 / fs
        self.beta = dt / (RC + dt)
        self.y = 0.0
    def __call__(self, x):
        y = self.beta * x + (1 - self.beta) * self.y
        self.y = y
        return y

# fc=0.3Hz è abbastanza bassa da NON distorcere PVC larghi (~150ms)
hp = HPFilter(fc=0.3, fs=SAMPLE_HZ)
lp = LPFilter(fc=25.0, fs=SAMPLE_HZ)

# buffer del segnale filtrato (centrato attorno a zero)
buf = collections.deque([0.0] * BUF_SIZE, maxlen=BUF_SIZE)

# state machine constants
FSM_IDLE   = 0  # sotto WIDTH_THR
FSM_WIDTH  = 1  # sopra WIDTH_THR ma sotto DETECT_THR
FSM_DETECT = 2  # sopra DETECT_THR (QRS confermato)
FSM_POST   = 3  # sotto WIDTH_THR dopo DETECT, monitoraggio rebound

state = {
    "ecg_bpm": 0,
    "sinus_bpm": 0,
    "pvc_rate": 0,
    "pvc_burden_pct": 0.0,
    "last_peak_sample": None,
    "samples_seen": 0,
    "rr_intervals": collections.deque(maxlen=12),
    "beats_window": collections.deque(),
    "peak_amplitudes": collections.deque(maxlen=30),
    "peaks": collections.deque(maxlen=200),
    "pvc_count": 0,
    "premature_count": 0,
    # state machine vars
    "fsm": FSM_IDLE,
    "current_peak_amp": 0.0,
    "current_peak_n": 0,
    "width_start": None,
    "pending_width_ms": 0.0,
    "pending_peak_n": 0,
    "pending_peak_amp": 0.0,
    "post_counter": 0,
    "post_trough": 0.0,
    "last_rebound_ratio": 0.0,
    "last_width_ms": 0.0,
    "lock": threading.Lock(),
}

def detect_threshold():
    """Soglia di rilevamento (alta): 45% della mediana ampiezze recenti.
    Bootstrap a DETECT_THR_FLOOR = 0.30V."""
    if len(state["peak_amplitudes"]) >= 3:
        med_amp = statistics.median(state["peak_amplitudes"])
        return max(DETECT_THR_FLOOR, med_amp * DETECT_THR_RATIO)
    return DETECT_THR_FLOOR

print(f"Avvio streamer su {DEVICE}...")
proc = subprocess.Popen(
    [MPREMOTE, "connect", DEVICE, "run", STREAMER],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    bufsize=1,
    universal_newlines=True,
)

def reader():
    REFRACTORY_SAMPLES = int(0.30 * SAMPLE_HZ)
    SETTLING_SAMPLES = int(1.5 * SAMPLE_HZ)
    POST_PEAK_SAMPLES = int(POST_PEAK_MS * SAMPLE_HZ / 1000)

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            v_raw = float(line)
        except ValueError:
            continue

        # filtro band-pass 0.3 - 25 Hz
        v = lp(hp(v_raw))

        with state["lock"]:
            buf.append(v)
            state["samples_seen"] += 1
            n = state["samples_seen"]

            if n < SETTLING_SAMPLES:
                continue

            dthr = detect_threshold()
            fsm = state["fsm"]

            # state machine: IDLE -> WIDTH -> DETECT -> POST -> IDLE
            if fsm == FSM_IDLE:
                if v > WIDTH_THR:
                    state["fsm"] = FSM_WIDTH
                    state["width_start"] = n
                    state["current_peak_amp"] = v
                    state["current_peak_n"] = n

            elif fsm == FSM_WIDTH:
                if v > state["current_peak_amp"]:
                    state["current_peak_amp"] = v
                    state["current_peak_n"] = n
                if v > dthr:
                    state["fsm"] = FSM_DETECT
                elif v < WIDTH_THR * 0.8:
                    # falso allarme (T-wave o rumore): non ha mai superato detect_thr
                    state["fsm"] = FSM_IDLE

            elif fsm == FSM_DETECT:
                if v > state["current_peak_amp"]:
                    state["current_peak_amp"] = v
                    state["current_peak_n"] = n
                if v < WIDTH_THR:
                    # fine corpo QRS, ora monitora rebound
                    state["pending_width_ms"] = (n - state["width_start"]) * 1000.0 / SAMPLE_HZ
                    state["pending_peak_n"]   = state["current_peak_n"]
                    state["pending_peak_amp"] = state["current_peak_amp"]
                    state["post_counter"] = 0
                    state["post_trough"] = v
                    state["fsm"] = FSM_POST

            elif fsm == FSM_POST:
                if v < state["post_trough"]:
                    state["post_trough"] = v
                state["post_counter"] += 1

                new_qrs = (v > WIDTH_THR and state["post_counter"] > REFRACTORY_SAMPLES)
                if state["post_counter"] >= POST_PEAK_SAMPLES or new_qrs:
                    p_n   = state["pending_peak_n"]
                    p_amp = state["pending_peak_amp"]
                    w_ms  = state["pending_width_ms"]
                    trough = state["post_trough"]
                    rebound = -trough if trough < 0 else 0.0
                    ratio = rebound / p_amp if p_amp > 0 else 0.0

                    state["last_width_ms"] = w_ms
                    state["last_rebound_ratio"] = ratio

                    lp_sample = state["last_peak_sample"]
                    if lp_sample is None or (p_n - lp_sample) > REFRACTORY_SAMPLES:
                        # CLASSIFICAZIONE: rebound OR width
                        classification = "normal"
                        if ratio >= REBOUND_RATIO_PVC or w_ms >= PVC_WIDTH_MS:
                            classification = "pvc"

                        if lp_sample is not None:
                            state["rr_intervals"].append((p_n - lp_sample) / SAMPLE_HZ)

                        state["peak_amplitudes"].append(p_amp)
                        if classification == "pvc":
                            state["pvc_count"] += 1
                        state["last_peak_sample"] = p_n
                        state["peaks"].append((p_n, p_amp, classification))

                        state["beats_window"].append((p_n, classification))
                        cutoff = n - BPM_WINDOW_S * SAMPLE_HZ
                        while state["beats_window"] and state["beats_window"][0][0] < cutoff:
                            state["beats_window"].popleft()

                        # calcolo dei tre BPM su finestra effettiva
                        elapsed_total_s = n / SAMPLE_HZ
                        effective_window_s = min(BPM_WINDOW_S, elapsed_total_s)
                        if effective_window_s >= 3.0:
                            beats = list(state["beats_window"])
                            n_all = len(beats)
                            n_norm = sum(1 for _, c in beats if c == "normal")
                            n_pvc_w = sum(1 for _, c in beats if c == "pvc")
                            state["ecg_bpm"]   = int(round(n_all  / effective_window_s * 60))
                            state["sinus_bpm"] = int(round(n_norm / effective_window_s * 60))
                            state["pvc_rate"]  = int(round(n_pvc_w / effective_window_s * 60))
                            state["pvc_burden_pct"] = 100.0 * n_pvc_w / max(1, n_all)

                    # transizione di stato post-finalizzazione
                    if new_qrs:
                        state["fsm"] = FSM_WIDTH
                        state["width_start"] = n
                        state["current_peak_amp"] = v
                        state["current_peak_n"] = n
                    else:
                        state["fsm"] = FSM_IDLE

threading.Thread(target=reader, daemon=True).start()

# --- UI ---
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#1e1e1e')
ax.set_facecolor('#0d0d0d')

x = np.linspace(-WINDOW_S, 0, BUF_SIZE)
(line_plot,) = ax.plot(x, list(buf), linewidth=1.0, color='#2ecc71')

normal_markers = ax.scatter([], [], s=60, marker='v', color='#2ecc71',
                            edgecolors='white', linewidths=0.5, zorder=5)
premature_markers = ax.scatter([], [], s=140, marker='v', color='#f39c12',
                               edgecolors='white', linewidths=1.0, zorder=6)
pvc_markers = ax.scatter([], [], s=220, marker='v', color='#e74c3c',
                         edgecolors='white', linewidths=1.0, zorder=7)

ax.set_xlim(-WINDOW_S, 0)
ax.set_ylim(-1.0, 1.5)
ax.set_xlabel("Time (s, scrolling)", color='#aaaaaa')
ax.set_ylabel("ECG filtered (V, centered)", color='#aaaaaa')
ax.set_title("ECG live — verde=normale, rosso=PVC (didattico)",
             color='#ffffff', fontsize=13)
ax.grid(True, alpha=0.2, color='#444444')
ax.tick_params(colors='#aaaaaa')
for spine in ax.spines.values():
    spine.set_color('#444444')

# tre numeri di BPM impilati a destra
ecg_label = ax.text(0.98, 0.96, "ECG", transform=ax.transAxes,
                    fontsize=10, va='top', ha='right', color='#888888')
ecg_text  = ax.text(0.98, 0.93, "-- BPM", transform=ax.transAxes,
                    fontsize=36, fontweight='bold', va='top', ha='right',
                    color='#2ecc71')

sinus_label = ax.text(0.98, 0.76, "sinus (≈ Garmin se PVC mancano)",
                      transform=ax.transAxes, fontsize=9, va='top', ha='right',
                      color='#888888')
sinus_text  = ax.text(0.98, 0.73, "-- BPM", transform=ax.transAxes,
                      fontsize=24, fontweight='bold', va='top', ha='right',
                      color='#3498db')

pvc_label = ax.text(0.98, 0.60, "PVC / pulse deficit", transform=ax.transAxes,
                    fontsize=9, va='top', ha='right', color='#888888')
pvc_text  = ax.text(0.98, 0.57, "-- /min", transform=ax.transAxes,
                    fontsize=20, fontweight='bold', va='top', ha='right',
                    color='#e74c3c')
burden_text = ax.text(0.98, 0.48, "", transform=ax.transAxes,
                      fontsize=12, va='top', ha='right', color='#e74c3c')

status_text = ax.text(0.02, 0.05, "", transform=ax.transAxes,
                      fontsize=10, color='#888888')

paused_text = ax.text(0.5, 0.5, "", transform=ax.transAxes,
                      fontsize=28, fontweight='bold', va='center', ha='center',
                      color='#f1c40f', alpha=0.9, zorder=20)

hint_text = ax.text(0.02, 0.95,
                    "[spazio]=pausa  [r]=reset  [↑↓]=zoom Y  [←→]=zoom X",
                    transform=ax.transAxes, fontsize=9, va='top', ha='left',
                    color='#666666')

# stato navigazione
nav = {"paused": False}

XLIM_DEFAULT = (-WINDOW_S, 0)
YLIM_DEFAULT = (-1.0, 1.5)

def on_key(event):
    if event.key == ' ':
        nav["paused"] = not nav["paused"]
        paused_text.set_text("⏸  PAUSED" if nav["paused"] else "")
        fig.canvas.draw_idle()
    elif event.key in ('r', 'R'):
        ax.set_xlim(*XLIM_DEFAULT)
        ax.set_ylim(*YLIM_DEFAULT)
        fig.canvas.draw_idle()
    elif event.key in ('up',):
        ymin, ymax = ax.get_ylim()
        rng = ymax - ymin
        c = (ymin + ymax) / 2
        ax.set_ylim(c - rng * 0.4, c + rng * 0.4)
        fig.canvas.draw_idle()
    elif event.key in ('down',):
        ymin, ymax = ax.get_ylim()
        rng = ymax - ymin
        c = (ymin + ymax) / 2
        ax.set_ylim(c - rng * 0.6, c + rng * 0.6)
        fig.canvas.draw_idle()
    elif event.key in ('left',):
        # piu secondi visibili (zoom out X) — mostra una porzione piu ampia centrata sull'attuale
        xmin, xmax = ax.get_xlim()
        rng = xmax - xmin
        c = (xmin + xmax) / 2
        ax.set_xlim(c - rng * 0.6, c + rng * 0.6)
        fig.canvas.draw_idle()
    elif event.key in ('right',):
        # meno secondi (zoom in X)
        xmin, xmax = ax.get_xlim()
        rng = xmax - xmin
        c = (xmin + xmax) / 2
        ax.set_xlim(c - rng * 0.4, c + rng * 0.4)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)

def update(frame):
    if nav["paused"]:
        # in pausa: niente aggiornamento, l'utente puo navigare liberamente
        return (line_plot, ecg_text, sinus_text, pvc_text, burden_text, status_text,
                normal_markers, premature_markers, pvc_markers, paused_text)
    with state["lock"]:
        ydata = list(buf)
        ecg_bpm = state["ecg_bpm"]
        sinus_bpm = state["sinus_bpm"]
        pvc_rate = state["pvc_rate"]
        pvc_burden = state["pvc_burden_pct"]
        n_seen = state["samples_seen"]
        n_window = len(state["beats_window"])
        n_pvc_total = state["pvc_count"]
        peaks_snapshot = list(state["peaks"])

    line_plot.set_ydata(ydata)
    ecg_text.set_text(f"{ecg_bpm if ecg_bpm else '--'} BPM")
    sinus_text.set_text(f"{sinus_bpm if sinus_bpm else '--'} BPM")
    if pvc_rate > 0:
        pvc_text.set_text(f"{pvc_rate}/min")
        burden_text.set_text(f"burden {pvc_burden:.1f}%")
    else:
        pvc_text.set_text("0/min")
        burden_text.set_text("")
    elapsed_s = n_seen / SAMPLE_HZ
    eff = min(BPM_WINDOW_S, int(elapsed_s))
    status_text.set_text(
        f"elapsed: {elapsed_s:.0f}s    BPM media su ultimi {eff}s "
        f"({n_window} battiti)    PVC totali: {n_pvc_total}"
    )

    normal_xy = []
    pvc_xy = []
    for (p_idx, p_amp, cls) in peaks_snapshot:
        x_rel = -(n_seen - p_idx) / SAMPLE_HZ
        if x_rel < -WINDOW_S:
            continue
        y = min(1.35, p_amp + 0.15)
        if cls == "pvc":
            pvc_xy.append((x_rel, y))
        else:
            normal_xy.append((x_rel, y))

    normal_markers.set_offsets(np.array(normal_xy) if normal_xy else np.empty((0, 2)))
    premature_markers.set_offsets(np.empty((0, 2)))
    pvc_markers.set_offsets(np.array(pvc_xy) if pvc_xy else np.empty((0, 2)))

    return (line_plot, ecg_text, sinus_text, pvc_text, burden_text, status_text,
            normal_markers, premature_markers, pvc_markers)

ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.tight_layout()

def cleanup():
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()

try:
    plt.show()
finally:
    cleanup()
