import RPi.GPIO as gp
from time import sleep

def dec2binary(value):
    return [int(bit) for bit in bin(value)[2:].zfill(8)]

def adc(comp, dac):
    for val in range(256):
        gp.output(dac, dec2binary(val))
        sleep(.01)
        compVal = gp.input(comp)
        if compVal == 1:
            return val

    return 256

gp.setmode(gp.BCM)

dac = [8, 11, 7, 1, 0, 5, 12, 6]
comp = 14
troyka = 13

gp.setup(dac, gp.OUT)
gp.setup(troyka, gp.OUT)
gp.setup(comp, gp.IN)


try:
    gp.output(troyka, 1)
    while True:
        out = adc(comp, dac)
        voltage = out / 256 * 3.3
        print(voltage, 'V')

finally:
    gp.output(troyka, 0)
    gp.output(dac, 0)
    gp.cleanup()
