import RPi.GPIO as gp
from time import sleep

def decimal2binary(value):
    return [int(bit) for bit in bin(value)[2:].zfill(8)]

def adc(comp, dac):
    binVal = [0 for i in range(8)]
    for i in range(8):
        binVal[i] = 1
        gp.output(dac, binVal)
        sleep(.01)
        compVal = gp.input(comp)
        if compVal == 1:
            binVal[i] = 0
    val = 0
    for i in range(8):
        val += binVal[i] * (2**(7-i))
    return val

def decimal2volume(val):
    volume = [0 for i in range(8)]
    step = 256 / 8
    for i in range(8):
        if val > step * i:
            volume[i] = 1
    return volume

gp.setmode(gp.BCM)

dac = [8, 11, 7, 1, 0, 5, 12, 6]
leds = [2, 3, 4, 17, 27, 22, 10, 9]
comp = 14
troyka = 13

gp.setup(dac, gp.OUT)
gp.setup(troyka, gp.OUT)
gp.setup(leds, gp.OUT)
gp.setup(comp, gp.IN)

try:
    gp.output(troyka, 1)
    while True:
        out = adc(comp, dac)
        gp.output(leds, decimal2volume(out))

finally:
    gp.output(troyka, 0)
    gp.output(dac, 0)
    gp.output(leds, 0)
    gp.cleanup()
