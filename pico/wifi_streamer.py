"""
WiFi streamer for Pi Pico W / Pico 2 W.

Connects to home WiFi, opens a TCP socket to the host running server.py,
and streams ADC samples (ECG) over the network at 250 Hz.

Drop-in replacement for streamer.py when the Pico is powered by battery
and disconnected from the host's USB.

Reads credentials from wifi_config.py (gitignored). Reconnects on failure.
"""
from machine import ADC, Pin
import network
import socket
import sys
import time

import wifi_config as cfg

# ---- ADC ----
adc       = ADC(Pin(26))    # GP26 / ADC0
SAMPLE_HZ = 250
PERIOD_US = 1_000_000 // SAMPLE_HZ
VREF      = 3.3
SCALE     = 65535

# ---- onboard LED (for status feedback) ----
try:
    led = Pin("LED", Pin.OUT)
except Exception:
    led = None

def set_led(state):
    if led is not None:
        try:
            led.value(state)
        except Exception:
            pass

def blink(n, t=0.1):
    for _ in range(n):
        set_led(1); time.sleep(t)
        set_led(0); time.sleep(t)

# ---- WiFi ----
def connect_wifi(timeout_s=30):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        return wlan
    print(f"WiFi: connecting to {cfg.SSID}...")
    wlan.connect(cfg.SSID, cfg.PASSWORD)
    t0 = time.ticks_ms()
    while not wlan.isconnected():
        if time.ticks_diff(time.ticks_ms(), t0) > timeout_s * 1000:
            print("WiFi: connect timed out")
            return None
        time.sleep(0.5)
    print("WiFi: connected,", wlan.ifconfig())
    return wlan

# ---- main loop ----
def stream_forever():
    wlan = None
    while True:
        # ensure WiFi
        if wlan is None or not wlan.isconnected():
            wlan = connect_wifi()
            if wlan is None:
                blink(3, 0.2)
                time.sleep(2)
                continue
        set_led(1)  # LED on while WiFi up

        # TCP connection
        s = None
        try:
            s = socket.socket()
            s.settimeout(5)
            print(f"TCP: connecting to {cfg.SERVER_IP}:{cfg.SERVER_PORT}...")
            s.connect((cfg.SERVER_IP, cfg.SERVER_PORT))
            s.settimeout(None)
            print("TCP: connected, streaming")
            blink(2, 0.05)
            set_led(1)

            # streaming loop @ 250 Hz
            t_next = time.ticks_us()
            buf = bytearray()
            while True:
                while time.ticks_diff(time.ticks_us(), t_next) < 0:
                    pass
                t_next = time.ticks_add(t_next, PERIOD_US)
                raw = adc.read_u16()
                volt = (raw / SCALE) * VREF
                line = ("%.4f\n" % volt).encode()
                # build a small batch (~25 samples = 100ms) before sending to reduce TCP overhead
                buf += line
                if len(buf) >= 200:           # ~25 lines
                    s.send(buf)
                    buf = bytearray()
        except Exception as e:
            print("TCP error:", e)
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass
            blink(4, 0.05)
            set_led(0)
            time.sleep(2)
            # loop continues and retries

stream_forever()
