"""
Blink test for Raspberry Pi Pico 2 W.
Lampeggia il LED onboard per 10 secondi a 1 Hz.
Conferma che MicroPython gira e abbiamo controllo del board.
"""
from machine import Pin
import time

led = Pin("LED", Pin.OUT)

for i in range(10):
    led.on()
    time.sleep(0.5)
    led.off()
    time.sleep(0.5)
    print(f"blink {i+1}/10")

print("blink test done")
