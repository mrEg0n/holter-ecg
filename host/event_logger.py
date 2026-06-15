"""
Event logger per gli esperimenti di PROVOCAZIONE (apnee, peso braccio, scale...).

Mentre il server (host/server.py) sta registrando, premi un tasto e questo
strumento manda un marker al server, che lo stampa col tempo ESATTO della
registrazione nel file  logs/markers_<sessione>.csv  (colonne t_s,text).
Cosi dopo, in analisi, puoi segmentare la traccia per manovra.

I tasti delle MANOVRE fanno da interruttore START/END (premi una volta = START,
ripremi = END), cosi ogni ripetizione resta delimitata. In alto vedi quali sono
"aperte". Ogni evento viene anche salvato in locale (logs/eventlog_<ora>.csv) con
l'orario da orologio, come backup nel caso il server non risponda.

USO:
    # 1) in un terminale, avvia la registrazione:
    TRANSPORT=usb python3 host/server.py      (oppure TRANSPORT=wifi)
    # 2) in un ALTRO terminale, avvia questo logger:
    python3 host/event_logger.py
    # (se il server e' su un altro PC:  python3 host/event_logger.py --url http://IP:8081 )

TASTI (manovre = interruttore START/END):
    b  baseline          f  apnea polmoni PIENI     e  apnea polmoni VUOTI
    l  peso braccio SX    r  peso braccio DX
    w  cammino            s  scale                   c  recupero
    spazio  marker istantaneo (es. i "3 colpetti" di sync / confine generico)
    t  testo libero        ?  mostra i tasti          q  esci
"""
import argparse
import json
import os
import sys
import termios
import tty
import urllib.request
from datetime import datetime

MANEUVERS = {  # tasto -> etichetta (interruttore start/end)
    "b": "baseline",
    "f": "apnea_full",      # polmoni pieni
    "e": "apnea_empty",     # polmoni vuoti
    "l": "weight_left",
    "r": "weight_right",
    "w": "walking",
    "s": "stairs",
    "c": "recovery",
}
HELP = __doc__.split("TASTI")[1]

def getch():
    """legge UN tasto senza invio (raw mode)."""
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

    open_state = {}   # etichetta -> True se "aperta" (in attesa di END)

    def post(text):
        """manda il marker al server; ritorna t_s (sec nella registrazione) o None."""
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
    print("EVENT LOGGER — registrazione esperimenti di provocazione")
    print(f"server: {args.url}   ·   backup locale: logs/eventlog_*.csv")
    print(HELP.rstrip())
    print("=" * 68)
    # verifica connessione
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
            # testo libero: torno in modalita' normale per leggere la riga
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
        # tasto non mappato: ignora
    backup.close()
    # avviso se restano manovre aperte (dimenticato l'END)
    leftover = [k for k, v in open_state.items() if v]
    if leftover:
        print(f"\n⚠ manovre lasciate APERTE (nessun END): {', '.join(leftover)}")
    print("\nChiuso. I marker sono in  logs/markers_<sessione>.csv  (e nel backup locale).")

if __name__ == "__main__":
    main()
