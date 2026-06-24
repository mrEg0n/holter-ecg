"""
Continuous ECG streamer from the Pico 2 W over USB serial.
Samples ADC0 (GP26) at 250 Hz and prints the value in Volts, one line per sample.
The dashboard on the Mac reads this stdout via mpremote.
"""
from machine import ADC, Pin
import time
import sys

adc = ADC(Pin(26))
SAMPLE_HZ = 250
PERIOD_US = 1_000_000 // SAMPLE_HZ
VREF = 3.3
SCALE = 65535

t_next = time.ticks_us()
while True:
    # busy-wait for tight timing
    while time.ticks_diff(time.ticks_us(), t_next) < 0:
        pass
    t_next = time.ticks_add(t_next, PERIOD_US)
    raw = adc.read_u16()
    volt = (raw / SCALE) * VREF
    sys.stdout.write(f"{volt:.4f}\n")
