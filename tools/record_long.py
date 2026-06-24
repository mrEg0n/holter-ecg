"""
Records 30 seconds of raw ECG at a nominal 250 Hz.
Saves CSV with the Pico's time_ms (real, derived from ticks_us) + raw + volt.
This way we can verify the Pico's actual sample rate.
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
