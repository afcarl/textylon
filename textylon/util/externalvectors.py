'''
Created on Feb 21, 2014

@author: af
'''
import codecs

from dsm_classes import Vocabulary, Matrix
import numpy as np


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
    CoocMat.makematrix(vocabSize, dimensionality)
    numRows = 0
    for line in lines:
        numRows += 1
        if theFirstTime:
            theFirstTime = False
            fs = line.split()
            vocabSize = int(fs[0])
            dimensionality = int(fs[1])
            continue
        fields = line.split()
        word = fields[0]
        l = fields[1:] 
        vector = np.array(map(float, l))
        if numRows == 2:
            matrix = vector
        else:
            matrix = np.vstack((matrix, vector))
        IndVoc.set(word, IndVoc.getlength())
        
    CoocMat.makeMatrixFromDense(matrix)
    return IndVoc, CoocMat

def loadExternalRepresentationRepLab(textFile, encoding='utf-8', dimension=100):
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
    wordID = 0
    with codecs.open(textFile, 'r', encoding) as f:
        for line in f:
            fields = line.split('\t')
            word = fields[0]
            IndVoc.set(word, wordID)
            wordID += 1
    
    wordID = 0       
    with codecs.open(textFile, 'r', encoding) as f:
        for line in f:
            fields = line.split('\t')
            word = fields[0]
            vector = map(float, fields[1].split(','))
            wordID += 1
        
    # the first line is the vocabulary size and the representation dimensionality
    lines = f.readlines()
    theFirstTime = True
    reprDict = {}
    vocabSize = 0
    dimensionality = 0
    matrix = None
    CoocMat.makematrix(vocabSize, dimensionality)
    numRows = 0
    for line in lines:
        numRows += 1
        if theFirstTime:
            theFirstTime = False
            fs = line.split()
            vocabSize = int(fs[0])
            dimensionality = int(fs[1])
            continue
        fields = line.split()
        word = fields[0]
        l = fields[1:] 
        vector = np.array(map(float, l))
        if numRows == 2:
            matrix = vector
        else:
            matrix = np.vstack((matrix, vector))
        IndVoc.set(word, IndVoc.getlength())
        
    CoocMat.makeMatrixFromDense(matrix)
    return IndVoc, CoocMat
