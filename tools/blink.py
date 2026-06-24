"""
Blink test for Raspberry Pi Pico 2 W.
Blinks the onboard LED for 10 seconds at 1 Hz.
Confirms that MicroPython is running and we have control of the board.
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
