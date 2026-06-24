"""
Event logger for the PROVOCATION experiments (breath-holds, arm load, stairs...).

While the server (host/server.py) is recording, press a key and this tool sends
a marker to the server, which stamps it with the EXACT recording time in the file
logs/markers_<session>.csv  (columns t_s,text).
This way, later in analysis, you can segment the trace by maneuver.

The MANEUVER keys act as a START/END toggle (press once = START, press again =
END), so each repetition stays delimited. At the top you can see which ones are
"open". Every event is also saved locally (logs/eventlog_<time>.csv) with the
wall-clock time, as a backup in case the server doesn't respond.

USAGE:
    # 1) in one terminal, start the recording:
    TRANSPORT=usb python3 host/server.py      (or TRANSPORT=wifi)
    # 2) in ANOTHER terminal, start this logger:
    python3 host/event_logger.py
    # (if the server is on another PC:  python3 host/event_logger.py --url http://IP:8081 )

KEYS (maneuvers = START/END toggle):
    b  baseline          f  breath-hold lungs FULL   e  breath-hold lungs EMPTY
    l  arm load LEFT      r  arm load RIGHT
    w  walking           s  stairs                   c  recovery
    space   instant marker (e.g. the "3 taps" sync / generic boundary)
    t  free text          ?  show the keys           q  quit
"""
import argparse
import json
import os
import sys
import termios
import tty
import urllib.request
from datetime import datetime

MANEUVERS = {  # key -> label (start/end toggle)
    "b": "baseline",
    "f": "apnea_full",      # lungs full
    "e": "apnea_empty",     # lungs empty
    "l": "weight_left",
    "r": "weight_right",
    "w": "walking",
    "s": "stairs",
    "c": "recovery",
}
HELP = __doc__.split("KEYS")[1]

def getch():
    """reads ONE key without Enter (raw mode)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8081",
                    help="indirizzo del server (default http://localhost:8081)")
    args = ap.parse_args()
    url = args.url.rstrip("/") + "/mark"

    os.makedirs("logs", exist_ok=True)
    backup = open(os.path.join("logs", f"eventlog_{datetime.now():%Y%m%d_%H%M%S}.csv"),
                  "w", buffering=1)
    backup.write("wall_time,text,server_t_s\n")

    open_state = {}   # label -> True if "open" (waiting for END)

    def post(text):
        """sends the marker to the server; returns t_s (sec into the recording) or None."""
        t_s = None
        try:
            req = urllib.request.Request(
                url, data=json.dumps({"text": text}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=3) as r:
                t_s = json.load(r).get("marker", {}).get("t_s")
        except Exception as ex:
            print(f"  ⚠ server non raggiunto ({ex}). Evento salvato solo in locale.")
        backup.write(f'{datetime.now():%H:%M:%S},"{text}",{("%.2f"%t_s) if t_s is not None else ""}\n')
        return t_s

    def show_open():
        opn = [k for k, v in open_state.items() if v]
        bar = ("  aperte ▶ " + ", ".join(opn)) if opn else "  (nessuna manovra aperta)"
        print(bar)

    print("=" * 68)
    print("EVENT LOGGER — logging of provocation experiments")
    print(f"server: {args.url}   ·   backup locale: logs/eventlog_*.csv")
    print(HELP.rstrip())
    print("=" * 68)
    # check connection
    try:
        urllib.request.urlopen(args.url.rstrip("/") + "/markers", timeout=3)
        print("✓ server raggiungibile. Premi i tasti durante gli esperimenti.\n")
    except Exception:
        print("⚠ ATTENZIONE: il server non risponde. Avvialo prima (host/server.py).")
        print("  Procedo comunque: gli eventi finiscono nel backup locale.\n")

    while True:
        ch = getch().lower()
        if ch in ("q", "\x03", "\x04"):           # q, Ctrl-C, Ctrl-D
            break
        if ch == "?":
            print(HELP.rstrip()); continue
        ts_label = datetime.now().strftime("%H:%M:%S")
        if ch == " ":
            t_s = post("sync"); show_t = f"  t={t_s:.1f}s" if t_s is not None else ""
            print(f"[{ts_label}] • sync{show_t}"); continue
        if ch == "t":
            # free text: switch back to normal mode to read the line
            sys.stdout.write("  testo > "); sys.stdout.flush()
            txt = sys.stdin.readline().strip()
            if txt:
                t_s = post(txt); show_t = f"  t={t_s:.1f}s" if t_s is not None else ""
                print(f"[{ts_label}] ✎ {txt}{show_t}")
            continue
        if ch in MANEUVERS:
            name = MANEUVERS[ch]
            is_open = open_state.get(name, False)
            phase = "END" if is_open else "START"
            t_s = post(f"{name} {phase}")
            open_state[name] = not is_open
            show_t = f"  t={t_s:.1f}s" if t_s is not None else ""
            arrow = "■" if phase == "END" else "▶"
            print(f"[{ts_label}] {arrow} {name} {phase}{show_t}")
            show_open()
            continue
        # unmapped key: ignore
    backup.close()
    # warn if any maneuvers are left open (forgot the END)
    leftover = [k for k, v in open_state.items() if v]
    if leftover:
        print(f"\n⚠ manovre lasciate APERTE (nessun END): {', '.join(leftover)}")
    print("\nClosed. Markers are in  logs/markers_<session>.csv  (and in the local backup).")

if __name__ == "__main__":
    main()
