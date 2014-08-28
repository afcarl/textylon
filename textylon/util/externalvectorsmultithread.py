'''
Created on Feb 21, 2014

@author: af
'''
import threading, Queue
import time

from dsm_classes import Vocabulary, Matrix
import numpy as np


class String2VectorThread(threading.Thread):
    def __init__(self, queue, IndVoc, outputQueue, lock):
        self.q = queue
        self.oq = outputQueue
        self.IndVoc = IndVoc
        self.lock = lock
    def run(self):
        while not self.q.empty():
            line = self.q.get()
            fields = line.split()
            word = fields[0]
            self.lock.acquire()
            l = self.IndVoc.getlength()
            self.IndVoc.set(word, l)
            self.lock.release()
            l = fields[1:] 
            vector = np.array(map(float, l))
            self.oq.put((l, vector))
    
    
def loadExternalRepresentation(textFile):
    """
    this function loads an external representation IndVoc and CoocMatrix
    the format of the file like this:
    first line is
    vocabulary_count dimensionality
    the other lines are of the follwoin format
    word value value .... value
    and the number of values is the dimensionality in the first line
    """
    IndVoc = Vocabulary()
    CoocMat = Matrix()
    
    f = open(textFile, 'r')
    # the first line is the vocabulary size and the representation dimensionality
    lines = f.readlines()
    theFirstTime = True
    reprDict = {}
    vocabSize = 0
    dimensionality = 0
    matrix = None
    q = Queue.Queue()
    threads = []    
    lock = threading.Lock()
    outQ = Queue.PriorityQueue()
    numRows = 0
    for line in lines:
        numRows += 1
        if theFirstTime:
            theFirstTime = False
            fs = line.split()
            vocabSize = int(fs[0])
            dimensionality = int(fs[1])
            continue
        q.put(line)

    for i in range(0, 20):
         threads.append(String2VectorThread(q, IndVoc, outQ, lock))
    while True:
        time.sleep(30)
        print ("%d is the size of the queue" % outQ.qsize())
        if outQ.empty():
            break
    print("building a matrix from indivisual rows...")
    FirstTime = True
    matrix = None
    while not outQ.empty():
        if FirstTime:
            matrix = outQ.get()
            FirstTime = False
        else:
            matrix = np.vstack(matrix, outQ.get())
    
    CoocMat.makeMatrixFromDense(matrix)
    print("finished building the matrix.")
    return IndVoc, CoocMat
