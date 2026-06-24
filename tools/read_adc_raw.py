"""
Raw read of ADC0 (GP26) from the Pico 2 W.
Polls 50 samples at 10 Hz, prints raw + voltage.
Sanity-check test without electrodes on the body.
"""
from machine import ADC, Pin
import time

adc = ADC(Pin(26))  # ADC0 / GP26

VREF = 3.3  # 3.3V reference
SCALE = 65535  # MicroPython exposes ADC as 16-bit (0..65535)

print("--- AD8232 raw read test ---")
print("sample | raw   | volt")
for i in range(50):
    raw = adc.read_u16()
    volt = (raw / SCALE) * VREF
    print(f"{i:3d}   | {raw:5d} | {volt:.3f} V")
    time.sleep(0.1)

print("--- done ---")
