'''
Created on Jan 10, 2014

@author: af
'''
import io

import matplotlib.pyplot as plot
import numpy as np


inputAddress = raw_input('input utf-8 file: ')
outputAddress = raw_input('output ascii file: ')
f = open(inputAddress, 'rb')
binaryContent = f.read()
f.close()
udata = binaryContent.decode("utf-8")
asciidata = udata.encode("ascii", "ignore")

f = open(outputAddress, 'w')
f.write(asciidata)
f.close()
