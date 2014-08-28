'''
Created on Jan 10, 2014

@author: af
'''
from debian.deb822 import OrderedSet
import io
from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# mpl.rcParams['legend.fontsize'] = 10
f = open('/home/af/sright/alog.log', 'r')
lines = f.readlines()
minFreqs = []
maxFreqs = []
windowSizes = []
toeflScores = []
removeMinFreq = True

for line in lines:
    fields = line.split()
    print fields
    if removeMinFreq:
        if(int(fields[0]) != 0):
            continue
    minFreqs.append(int(fields[0]))
    maxFreqs.append(int(fields[1]))
    windowSizes.append(int(fields[2]))
    toeflScores.append(float(fields[3]))



if removeMinFreq:
    pass
else:
    plt.plot(minFreqs, toeflScores, 'bs')
    plt.xlabel("lower bound frequency threshold")
    plt.ylabel("toefl synonym test score")
    plt.show()

threedimensional = False
if threedimensional:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.plot(minFreqs, toeflScores, 'r--', maxFreqs, toeflScores, 'bs')
    ax.plot(windowSizes, maxFreqs, toeflScores, label='parametric curve')
    ax.legend()
    plt.show()

fig = plt.figure()
fig.suptitle('TOEFL test score using right contexts and PPMI weighting', fontsize=20)
multipleMaxFreq = True
if multipleMaxFreq:
    maxFreqSet = OrderedSet(maxFreqs)
    j = 1
    for max in maxFreqSet:
        if max < 5000 :
            continue
        windows = [w for w, m in zip(windowSizes, maxFreqs) if m == max]  
        scores = [s for s, m in zip(toeflScores, maxFreqs) if m == max]
        ax = plt.subplot(len(maxFreqSet), 2, j)
        j += 1
        ax.plot(windows, scores, 'bo-')
        ax.set_title('maxFreq = ' + str(max))
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize('small')
        ax.set_xlabel('window size')
        ax.set_ylabel('TOEFL score')
        
        
        # plt.ylabel('toefl score')
        # plt.xlabel('window size')
    
        
# plt.tight_layout()            
plt.show()        
if 1 == 1:
    exit(0)

x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)


plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'yo-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.show()
plt.show()
