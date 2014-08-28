'''
Created on Jan 28, 2014
This class converts a corpus into features extracted from a distributinal semantic model. Given the representations of each word and the vocabulary
this class summes the representation of every work in a document and builds a DSM representation for that document.
@author: af
'''

from scipy.sparse import vstack, hstack
from scipy.sparse.csr import csr_matrix

import numpy as np


class DSMVectorizer():
    def __init__(self):
        pass
                
                
                

    def fit_transform(self, raw_documents, dsmMatrix, vocabulary):
        """use the sum of DSM vectors of a document's words as that document vector.
        here dsmMatrix is the matrix used in DSM that contains the distributional representations of all vocabulary words.
        """
        x = None
        i = 0
        num_exception = 0
        docnumber = 0
        docMatrices = []
        oov = 0
        oovDoc = 0
        for doc in raw_documents:
            if docnumber % 1000 == 0:
                print "processing document number " + str(docnumber + 1)
            docnumber += 1
            if oov > docnumber * 5:
                print "warning: " + str(oov) + " oov."
            docMatrix = None
            words = doc.split()
            for word in words:
                wordId = vocabulary.getindex(word)
                if not wordId:
                    oov += 1
                    continue
                # doc matrix is the sum of all its word vectors
                if docMatrix is not None:
                    docMatrix = docMatrix + dsmMatrix.getSparseRow(wordId)
                else:
                    docMatrix = dsmMatrix.getSparseRow(wordId)
            if docMatrix is None:
                print dsmMatrix.matrix.shape
                print "%d is the shape of dsmMatrix" % dsmMatrix.matrix.shape[1]
                docMatrix = csr_matrix((1, dsmMatrix.matrix.shape[1]))
                docMatrix[0, 0] = 0
                oovDoc += 1
            docMatrices.append(docMatrix)
        print "%d documents are completely out of vocabulary" % oovDoc
        print "vstacking matrices..."
        x = vstack(docMatrices)
        print "vstacking finished."
        return x

class DSMSequenceVectorizer():
    def __init__(self):
        pass
                
                
                

    def fit_transform(self, raw_documents, dsmMatrix, vocabulary):
        """use the sum of DSM vectors of a document's words as that document vector.
        here dsmMatrix is the matrix used in DSM that contains the distributional representations of all vocabulary words.
        """
        x = None
        i = 0
        num_exception = 0
        docnumber = 0
        docMatrices = []
        oov = 0
        oovDoc = 0
        for doc in raw_documents:
            if docnumber % 1000 == 0:
                print "processing document number " + str(docnumber + 1)
            docnumber += 1
            if oov > docnumber:
                print "warning: " + str(oov) + " oov."
            docMatrix = None
            sequenceVectors = []
            words = doc.split()
            numAddedWords = 0
            for word in words:
                wordId = vocabulary.getindex(word)
                if not wordId:
                    oov += 1
                    continue
                # doc matrix is the sum of all its word vectors
                wordRepresentation = dsmMatrix.getSparseRow(wordId)
                sequenceVectors.append(wordRepresentation)
                numAddedWords += 1
                if numAddedWords == 20:
                    break
            if numAddedWords == 0:
                oovDoc += 1
            # if text is shorter than 20 words add some extra zero vectors
            while numAddedWords < 20:
                zeroSparse = csr_matrix((1, dsmMatrix.matrix.shape[1]))
                sequenceVectors.append(zeroSparse)
                numAddedWords += 1
            # hstack word vectors in sequence
            docMatrix = hstack(sequenceVectors)
            docMatrices.append(docMatrix)
        print "%d documents are completely out of vocabulary" % oovDoc
        print "vstacking matrices..."
        x = vstack(docMatrices)
        print "vstacking finished."
        return x
