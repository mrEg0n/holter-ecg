"""
Lettura grezza ADC0 (GP26) dal Pico 2 W.
Polla 50 campioni a 10 Hz, stampa raw + voltage.
Test di sanity check senza elettrodi sul corpo.
"""
from machine import ADC, Pin
import time

adc = ADC(Pin(26))  # ADC0 / GP26

VREF = 3.3  # 3.3V reference
SCALE = 65535  # MicroPython espone ADC come 16-bit (0..65535)

print("--- AD8232 raw read test ---")
print("sample | raw   | volt")
for i in range(50):
    raw = adc.read_u16()
    volt = (raw / SCALE) * VREF
    print(f"{i:3d}   | {raw:5d} | {volt:.3f} V")
    time.sleep(0.1)

print("--- done ---")
