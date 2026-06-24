"""
Web dashboard for live ECG monitoring via SSE on port 8081.

Reuses the same pipeline as host/dashboard.py:
  - IIR band-pass (HP 0.3 Hz + LP 25 Hz)
  - 4-state QRS detector (IDLE -> WIDTH -> DETECT -> POST)
  - PVC classification by rebound ratio + width

Adds:
  - Flask HTTP server with Server-Sent Events for live data
  - CSV logging to ../logs/ecg_YYYYMMDD_HHMMSS.csv  (every sample)
  - CSV logging to ../logs/peaks_YYYYMMDD_HHMMSS.csv (every detected beat)

Open http://localhost:8081 in a browser (also reachable from any device on the LAN).
Educational use only — not a medical device.
"""
import collections
import glob
import json
import math
import os
import queue
import shutil
import socket
import statistics
import subprocess
import threading
import time
from datetime import datetime

from flask import Flask, Response, request, send_from_directory

HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(HERE)
LOG_DIR   = os.path.join(ROOT, "logs")
STATIC    = os.path.join(HERE, "static")
PORT      = int(os.environ.get("PORT", "8081"))
TRANSPORT = os.environ.get("TRANSPORT", "usb").lower()   # "usb" | "wifi"
TCP_PORT  = int(os.environ.get("TCP_PORT", "5005"))      # TCP port for Pico → server stream
os.makedirs(LOG_DIR, exist_ok=True)


# ---------------- mpremote / device autodetect ----------------
def find_mpremote():
    if shutil.which("mpremote"):
        return "mpremote"
    home = os.path.expanduser("~")
    for pat in ("Library/Python/*/bin/mpremote", ".local/bin/mpremote"):
        c = glob.glob(os.path.join(home, pat))
        if c:
            return c[0]
    raise SystemExit("install with: pip3 install --user mpremote")

def find_pico_device():
    env = os.environ.get("PICO_DEVICE")
    if env:
        return env
    for pat in ("/dev/cu.usbmodem*", "/dev/ttyACM*"):
        c = sorted(glob.glob(pat))
        if c:
            return c[0]
    raise SystemExit("Pico not detected. Plug USB or set PICO_DEVICE=/dev/...")

if TRANSPORT == "usb":
    MPREMOTE = find_mpremote()
    DEVICE   = find_pico_device()
    STREAMER = os.path.normpath(os.path.join(ROOT, "pico", "streamer.py"))
else:
    MPREMOTE = DEVICE = STREAMER = None
    print(f"[transport] WIFI mode: TCP server on 0.0.0.0:{TCP_PORT}")


# ---------------- DSP / detector constants ----------------
SAMPLE_HZ          = 250
WINDOW_S           = 180   # 3 minutes of scrolling trace
BUF_SIZE           = SAMPLE_HZ * WINDOW_S
WIDTH_THR          = 0.10
DETECT_THR_FLOOR   = 0.30
DETECT_THR_RATIO   = 0.45
POST_PEAK_MS       = 200
REBOUND_RATIO_PVC  = 0.40
PVC_WIDTH_MS       = 95.0
PVC_MIN_AMP_V      = 0.70   # minimum amplitude required to accept a PVC
                            # (excludes small noisy normal beats)
REFRACTORY_S       = 0.30
BPM_WINDOW_S       = 60

REFRACTORY_SAMPLES = int(REFRACTORY_S * SAMPLE_HZ)
SETTLING_SAMPLES   = int(1.5 * SAMPLE_HZ)
POST_PEAK_SAMPLES  = int(POST_PEAK_MS * SAMPLE_HZ / 1000)

FSM_IDLE, FSM_WIDTH, FSM_DETECT, FSM_POST = 0, 1, 2, 3


class HPFilter:
    def __init__(self, fc, fs):
        RC = 1.0 / (2 * math.pi * fc); dt = 1.0 / fs
        self.alpha = RC / (RC + dt); self.y = 0.0; self.x = 0.0
    def __call__(self, x):
        y = self.alpha * (self.y + x - self.x)
        self.y = y; self.x = x
        return y

class LPFilter:
    def __init__(self, fc, fs):
        RC = 1.0 / (2 * math.pi * fc); dt = 1.0 / fs
        self.beta = dt / (RC + dt); self.y = 0.0
    def __call__(self, x):
        y = self.beta * x + (1 - self.beta) * self.y
        self.y = y
        return y

hp = HPFilter(0.3, SAMPLE_HZ)
lp = LPFilter(25.0, SAMPLE_HZ)


# ---------------- shared state (guarded by state["lock"]) ----------------
state = {
    "samples_seen":    0,
    "fsm":             FSM_IDLE,
    "current_peak_amp": 0.0,
    "current_peak_n":  0,
    "width_start":     None,
    "pending_width_ms": 0.0,
    "pending_peak_n":  0,
    "pending_peak_amp": 0.0,
    "post_counter":    0,
    "post_trough":     0.0,
    "last_peak_sample": None,
    "peak_amplitudes": collections.deque(maxlen=30),
    "beats_window":    collections.deque(),
    "ecg_bpm":         0,
    "sinus_bpm":       0,
    "pvc_rate":        0,
    "pvc_burden_pct":  0.0,
    "pvc_count_total": 0,
    "normal_count_total": 0,
    # SSE batching: cleared every time /stream pulls a batch
    "first_sample_of_session_t": None,
    "markers":         [],     # list of {n, t_s, text}
    "markers_revision": 0,     # incremented on every marker change
    "lock":            threading.Lock(),
}

# One queue per connected SSE client. Each client has its own separate queue
# so that two browsers open at the same time don't steal samples from each other.
clients = []          # list of {samples_q, peaks_q, last_markers_rev}
clients_lock = threading.Lock()
MAX_QUEUE = 4000      # if a client is too slow, drop the extra samples


def detect_threshold():
    pa = state["peak_amplitudes"]
    if len(pa) >= 3:
        return max(DETECT_THR_FLOOR, statistics.median(pa) * DETECT_THR_RATIO)
    return DETECT_THR_FLOOR


# ---------------- logging ----------------
session_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
samples_path   = os.path.join(LOG_DIR, f"ecg_{session_id}.csv")
peaks_path     = os.path.join(LOG_DIR, f"peaks_{session_id}.csv")
markers_path   = os.path.join(LOG_DIR, f"markers_{session_id}.csv")
samples_log    = open(samples_path, "w", buffering=1)
peaks_log      = open(peaks_path,   "w", buffering=1)
markers_log    = open(markers_path, "w", buffering=1)
samples_log.write("t_s,raw,filt\n")
peaks_log.write("t_s,amp_V,width_ms,rebound_ratio,class\n")
markers_log.write("t_s,text\n")
print(f"[log] samples: {samples_path}")
print(f"[log] peaks:   {peaks_path}")
print(f"[log] markers: {markers_path}")


# ---------------- per-sample pipeline ----------------
def process_sample(v_raw, v_filt):
    state["samples_seen"] += 1
    n = state["samples_seen"]
    t_s = n / SAMPLE_HZ

    # log every sample (raw + filtered)
    samples_log.write(f"{t_s:.4f},{v_raw:.4f},{v_filt:.4f}\n")

    # push to every connected SSE client (each has its own queue)
    with clients_lock:
        for c in clients:
            try:
                c["samples_q"].put_nowait(v_filt)
            except queue.Full:
                pass

    if n < SETTLING_SAMPLES:
        return

    v   = v_filt
    dthr = detect_threshold()
    fsm = state["fsm"]

    if fsm == FSM_IDLE:
        if v > WIDTH_THR:
            state["fsm"] = FSM_WIDTH
            state["width_start"] = n
            state["current_peak_amp"] = v
            state["current_peak_n"]   = n

    elif fsm == FSM_WIDTH:
        if v > state["current_peak_amp"]:
            state["current_peak_amp"] = v; state["current_peak_n"] = n
        if v > dthr:
            state["fsm"] = FSM_DETECT
        elif v < WIDTH_THR * 0.8:
            state["fsm"] = FSM_IDLE

    elif fsm == FSM_DETECT:
        if v > state["current_peak_amp"]:
            state["current_peak_amp"] = v; state["current_peak_n"] = n
        if v < WIDTH_THR:
            state["pending_width_ms"] = (n - state["width_start"]) * 1000.0 / SAMPLE_HZ
            state["pending_peak_n"]   = state["current_peak_n"]
            state["pending_peak_amp"] = state["current_peak_amp"]
            state["post_counter"]     = 0
            state["post_trough"]      = v
            state["fsm"]              = FSM_POST

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
            lps = state["last_peak_sample"]
            if lps is None or (p_n - lps) > REFRACTORY_SAMPLES:
                # PVC if deep rebound OR wide QRS, AND amplitude above the minimum threshold
                # (the AND with amp avoids mistaking small noisy normal beats with a
                # relatively tall physiological S-wave for PVCs)
                is_pvc_shape = (ratio >= REBOUND_RATIO_PVC or w_ms >= PVC_WIDTH_MS)
                cls = "pvc" if (is_pvc_shape and p_amp >= PVC_MIN_AMP_V) else "normal"
                state["peak_amplitudes"].append(p_amp)
                state["last_peak_sample"] = p_n
                if cls == "pvc":
                    state["pvc_count_total"] += 1
                else:
                    state["normal_count_total"] += 1
                state["beats_window"].append((p_n, cls))
                cutoff = n - BPM_WINDOW_S * SAMPLE_HZ
                while state["beats_window"] and state["beats_window"][0][0] < cutoff:
                    state["beats_window"].popleft()

                peak_t = p_n / SAMPLE_HZ
                peaks_log.write(f"{peak_t:.4f},{p_amp:.4f},{w_ms:.1f},{ratio:.3f},{cls}\n")
                peak_payload = {
                    "n":        p_n,
                    "amp":      round(p_amp, 4),
                    "width_ms": round(w_ms, 1),
                    "rebound":  round(ratio, 3),
                    "cls":      cls,
                }
                with clients_lock:
                    for c in clients:
                        try:
                            c["peaks_q"].put_nowait(peak_payload)
                        except queue.Full:
                            pass

                elapsed = n / SAMPLE_HZ
                effw    = min(BPM_WINDOW_S, elapsed)
                if effw >= 3.0:
                    bw = state["beats_window"]
                    n_all  = len(bw)
                    n_norm = sum(1 for _, c in bw if c == "normal")
                    n_pvc  = n_all - n_norm
                    state["ecg_bpm"]        = round(n_all  / effw * 60)
                    state["sinus_bpm"]      = round(n_norm / effw * 60)
                    state["pvc_rate"]       = round(n_pvc  / effw * 60)
                    state["pvc_burden_pct"] = 100.0 * n_pvc / max(1, n_all)

            if new_qrs:
                state["fsm"] = FSM_WIDTH
                state["width_start"]     = n
                state["current_peak_amp"] = v
                state["current_peak_n"]  = n
            else:
                state["fsm"] = FSM_IDLE


# ---------------- Pico reader threads (USB or WiFi) ----------------
def process_line(line: str):
    """Parse a single sample line from the Pico and run the detector pipeline."""
    line = line.strip()
    if not line:
        return
    try:
        v_raw = float(line)
    except ValueError:
        return
    v_filt = lp(hp(v_raw))
    with state["lock"]:
        process_sample(v_raw, v_filt)


def usb_reader_loop():
    while True:
        try:
            print(f"[pico-usb] starting streamer on {DEVICE}")
            proc = subprocess.Popen(
                [MPREMOTE, "connect", DEVICE, "run", STREAMER],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=1, universal_newlines=True,
            )
            for line in proc.stdout:
                process_line(line)
        except Exception as e:
            print(f"[pico-usb] error: {e}")
        time.sleep(2)
        print("[pico-usb] reconnecting...")


def wifi_reader_loop():
    """TCP server: accepts one connection from the Pico WiFi streamer at a time.

    Uses a recv() timeout so that abrupt disconnects (power-off, WiFi drop) are
    detected within ~3 seconds and the server can accept the next connection
    instead of staying stuck on a dead socket.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", TCP_PORT))
    srv.listen(1)
    print(f"[pico-wifi] listening on 0.0.0.0:{TCP_PORT}")
    RECV_TIMEOUT_S = 3.0   # any silence longer than this = client is gone
    while True:
        try:
            conn, addr = srv.accept()
            print(f"[pico-wifi] connected from {addr}")
            conn.settimeout(RECV_TIMEOUT_S)
            buf = b""
            with conn:
                while True:
                    try:
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        print(f"[pico-wifi] silent for {RECV_TIMEOUT_S:.0f}s, dropping client {addr}")
                        break
                    if not chunk:
                        print("[pico-wifi] client closed")
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        process_line(line.decode("ascii", errors="ignore"))
        except Exception as e:
            print(f"[pico-wifi] error: {e}")
        time.sleep(0.5)


if TRANSPORT == "usb":
    threading.Thread(target=usb_reader_loop, daemon=True).start()
else:
    threading.Thread(target=wifi_reader_loop, daemon=True).start()


# ---------------- Flask app ----------------
app = Flask(__name__, static_folder=STATIC)

@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")

@app.route("/marker", methods=["POST"])
def add_marker():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    n    = data.get("n")
    if not text or n is None:
        return {"error": "missing text or n"}, 400
    try:
        n = int(n)
    except (TypeError, ValueError):
        return {"error": "invalid n"}, 400
    text = text[:200]
    marker = {"n": n, "t_s": n / SAMPLE_HZ, "text": text}
    with state["lock"]:
        state["markers"].append(marker)
        state["markers_revision"] += 1
    escaped = text.replace('"', '""')
    markers_log.write(f'{marker["t_s"]:.4f},"{escaped}"\n')
    return {"ok": True, "marker": marker}

@app.route("/mark", methods=["POST"])
def add_mark_now():
    """Like /marker, but the server stamps the marker with the CURRENT sample: the
    client sends only {text} (used by host/event_logger.py for the provocation
    experiments). This way the time is exact on the recording's clock."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return {"error": "missing text"}, 400
    text = text[:200]
    with state["lock"]:
        n = int(state.get("samples_seen", 0))
        marker = {"n": n, "t_s": n / SAMPLE_HZ, "text": text}
        state["markers"].append(marker)
        state["markers_revision"] += 1
    escaped = text.replace('"', '""')
    markers_log.write(f'{marker["t_s"]:.4f},"{escaped}"\n')
    return {"ok": True, "marker": marker}

@app.route("/markers", methods=["GET"])
def list_markers():
    with state["lock"]:
        return {"markers": list(state["markers"])}

@app.route("/stream")
def stream():
    # Each SSE client gets its own dedicated queue.
    # No competition between browsers over samples.
    client = {
        "samples_q":        queue.Queue(maxsize=MAX_QUEUE),
        "peaks_q":          queue.Queue(maxsize=MAX_QUEUE),
        "last_markers_rev": -1,
    }
    with clients_lock:
        clients.append(client)

    def gen():
        try:
            hello = {"hello": True, "fs": SAMPLE_HZ, "window_s": WINDOW_S,
                     "session_id": session_id}
            yield f"data: {json.dumps(hello)}\n\n"
            # initial markers
            with state["lock"]:
                init_markers = list(state["markers"])
                client["last_markers_rev"] = state["markers_revision"]
            yield f"data: {json.dumps({'markers': init_markers})}\n\n"

            while True:
                time.sleep(0.1)
                # drain queues per-client
                samples = []
                try:
                    while True:
                        samples.append(client["samples_q"].get_nowait())
                except queue.Empty:
                    pass
                peaks = []
                try:
                    while True:
                        peaks.append(client["peaks_q"].get_nowait())
                except queue.Empty:
                    pass

                with state["lock"]:
                    stats = {
                        "ecg_bpm":         state["ecg_bpm"],
                        "sinus_bpm":       state["sinus_bpm"],
                        "pvc_rate":        state["pvc_rate"],
                        "pvc_burden_pct":  round(state["pvc_burden_pct"], 1),
                        "samples_seen":    state["samples_seen"],
                        "pvc_total":       state["pvc_count_total"],
                        "normal_total":    state["normal_count_total"],
                    }
                    markers_payload = None
                    if state["markers_revision"] != client["last_markers_rev"]:
                        markers_payload = list(state["markers"])
                        client["last_markers_rev"] = state["markers_revision"]

                if samples or peaks or markers_payload is not None:
                    data = {"samples": samples, "peaks": peaks, "stats": stats}
                    if markers_payload is not None:
                        data["markers"] = markers_payload
                    yield f"data: {json.dumps(data)}\n\n"
        finally:
            with clients_lock:
                if client in clients:
                    clients.remove(client)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    print(f"[server] http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False, use_reloader=False)
