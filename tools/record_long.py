"""
Registra 30 secondi di ECG raw a 250 Hz nominali.
Salva CSV con time_ms del Pico (real, derivato da ticks_us) + raw + volt.
Cosi possiamo verificare il sample rate reale del Pico.
"""
from machine import ADC, Pin
import time
import sys

adc = ADC(Pin(26))
SAMPLE_HZ = 250
DURATION_S = 30
N_SAMPLES = SAMPLE_HZ * DURATION_S
PERIOD_US = 1_000_000 // SAMPLE_HZ

VREF = 3.3
SCALE = 65535

print("# t_us_pico,raw,volt")
t_start = time.ticks_us()
t_next = t_start
for i in range(N_SAMPLES):
    while time.ticks_diff(time.ticks_us(), t_next) < 0:
        pass
    t_next = time.ticks_add(t_next, PERIOD_US)

    raw = adc.read_u16()
    t_us = time.ticks_diff(time.ticks_us(), t_start)
    volt = (raw / SCALE) * VREF
    sys.stdout.write(f"{t_us},{raw},{volt:.4f}\n")

print("# done")
