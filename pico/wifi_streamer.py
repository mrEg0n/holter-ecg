"""
WiFi streamer multi-rete per Pi Pico W / Pico 2 W.

Al boot esegue una scansione WiFi e si connette alla prima rete configurata
in `wifi_config.NETWORKS` che risulta visibile. Per ogni rete è definito
anche l'IP del server LAN da contattare via TCP, così il dispositivo si
adatta automaticamente quando viene spostato fra case con LAN diverse.

Streama poi i campioni ADC (ECG) a 250 Hz sul server. Reconnect in caso
di drop. Pensato per essere salvato come main.py su flash per autostart.
"""
from machine import ADC, Pin
import network
import socket
import sys
import time

import wifi_config as cfg

# ---- ADC ----
adc       = ADC(Pin(26))
SAMPLE_HZ = 250
PERIOD_US = 1_000_000 // SAMPLE_HZ
VREF      = 3.3
SCALE     = 65535

# ---- LED status feedback ----
try:
    led = Pin("LED", Pin.OUT)
except Exception:
    led = None

def set_led(state):
    if led is not None:
        try: led.value(state)
        except Exception: pass

def blink(n, t=0.1):
    for _ in range(n):
        set_led(1); time.sleep(t)
        set_led(0); time.sleep(t)

# ---- WiFi multi-rete ----
def scan_wifi(wlan):
    """Restituisce un set di SSID visibili."""
    try:
        return {ssid.decode() if isinstance(ssid, bytes) else str(ssid)
                for ssid, *_ in wlan.scan()}
    except Exception as e:
        print("scan error:", e)
        return set()

def connect_one(wlan, ssid, password, timeout_s=20):
    """Prova a connettersi a una rete specifica. Ritorna True/False."""
    print(f"WiFi: connecting to {ssid}...")
    try: wlan.disconnect()
    except Exception: pass
    time.sleep(0.5)
    wlan.connect(ssid, password)
    t0 = time.ticks_ms()
    while not wlan.isconnected():
        if time.ticks_diff(time.ticks_ms(), t0) > timeout_s * 1000:
            print(f"WiFi: timeout su {ssid}, status={wlan.status()}")
            return False
        time.sleep(0.5)
    print(f"WiFi: connected to {ssid}, ifconfig={wlan.ifconfig()}")
    return True

def connect_any():
    """Cerca la prima rete configurata visibile, prova a connettersi.
    Ritorna (wlan, server_ip) oppure (None, None) se nessuna disponibile."""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    time.sleep(0.5)
    visible = scan_wifi(wlan)
    print(f"WiFi visibili ({len(visible)}): {sorted(visible)[:10]}...")
    for net in cfg.NETWORKS:
        if net["ssid"] in visible:
            if connect_one(wlan, net["ssid"], net["password"]):
                return wlan, net["server_ip"]
        else:
            print(f"WiFi: '{net['ssid']}' non visibile, skip")
    # fallback: prova a connettersi anche alle reti non viste dallo scan
    # (può succedere che lo scan abbia mancato la rete in quel momento)
    print("WiFi: nessuna rete visibile, tento i fallback...")
    for net in cfg.NETWORKS:
        if connect_one(wlan, net["ssid"], net["password"], timeout_s=12):
            return wlan, net["server_ip"]
    return None, None

# ---- main loop ----
def stream_forever():
    wlan = None
    server_ip = None
    while True:
        if wlan is None or not wlan.isconnected():
            wlan, server_ip = connect_any()
            if wlan is None:
                print("WiFi: nessuna rete raggiungibile, retry tra 5s")
                blink(3, 0.2)
                time.sleep(5)
                continue
        set_led(1)

        s = None
        try:
            s = socket.socket()
            s.settimeout(5)
            print(f"TCP: connecting to {server_ip}:{cfg.SERVER_PORT}...")
            s.connect((server_ip, cfg.SERVER_PORT))
            s.settimeout(None)
            print("TCP: connected, streaming")
            blink(2, 0.05); set_led(1)

            t_next = time.ticks_us()
            buf = bytearray()
            while True:
                while time.ticks_diff(time.ticks_us(), t_next) < 0:
                    pass
                t_next = time.ticks_add(t_next, PERIOD_US)
                raw = adc.read_u16()
                volt = (raw / SCALE) * VREF
                line = ("%.4f\n" % volt).encode()
                buf += line
                if len(buf) >= 200:
                    s.send(buf)
                    buf = bytearray()
        except Exception as e:
            print("TCP/stream error:", e)
            try:
                if s is not None: s.close()
            except Exception: pass
            blink(4, 0.05); set_led(0)
            # se l'errore è di rete, mantieni wlan; se persistente, forzeremo riconnessione
            time.sleep(2)
            # se siamo ancora connessi al WiFi, riprova solo il TCP — altrimenti tutto da capo
            if wlan is not None and not wlan.isconnected():
                wlan = None
                server_ip = None

stream_forever()
