from gpiozero import Device
from gpiozero.pins.lgpio import LGPIOFactory   # ← requires `lgpio` package
Device.pin_factory = LGPIOFactory()


from gpiozero import LED
from time import sleep

led = LED(2)

while True:
    led.on()
    sleep(1)
    led.off()
    sleep(1)