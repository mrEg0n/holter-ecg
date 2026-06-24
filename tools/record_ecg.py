"""
Records 5 seconds of ECG at 250 Hz from the Pico 2 W.
Prints all samples as CSV over serial: time_ms,raw,volt
"""
from machine import ADC, Pin
import time

adc = ADC(Pin(26))  # ADC0 / GP26
SAMPLE_HZ = 250
DURATION_S = 5
N_SAMPLES = SAMPLE_HZ * DURATION_S
PERIOD_US = 1_000_000 // SAMPLE_HZ  # 4000 us = 4 ms

VREF = 3.3
SCALE = 65535

print("# t_ms,raw,volt")
t_start = time.ticks_us()
t_next = t_start
for i in range(N_SAMPLES):
    # busy-wait for precise timing
    while time.ticks_diff(time.ticks_us(), t_next) < 0:
        pass
    t_next = time.ticks_add(t_next, PERIOD_US)

    raw = adc.read_u16()
    t_ms = (time.ticks_diff(time.ticks_us(), t_start)) / 1000
    volt = (raw / SCALE) * VREF
    print(f"{t_ms:.1f},{raw},{volt:.4f}")

print("# done")
