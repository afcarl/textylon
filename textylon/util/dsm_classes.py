import math
from numpy import log
import sklearn.decomposition
from sklearn.feature_extraction.text import TfidfTransformer
from sparsesvd import sparsesvd

import numpy as np
import scipy.sparse as sc


# vocabularies
class Vocabulary:

    def __init__(self):
        self.hsh = {}

    def getindex(self, w):
        if w in self.hsh:
            return self.hsh[w]
        else:
            return False

    def set(self, w, new):
        self.hsh[w] = new

    def getlength(self):
        return len(self.hsh)

    def lookup(self, w):
        if w in self.hsh:
            return True
        else:
            return False

    def delete(self):
        self.hsh = {}

# cooccurrence matrix
class Matrix:

    def makematrix(self, rows, cols):
        self.matrix = sc.lil_matrix((rows, cols))
    def makeMatrixFromDense(self, mat):
        self.matrix = sc.lil_matrix(mat)

    def update(self, row, col, val):
        self.matrix[row, col] += val

    def getvector(self, wordind):
        return self.matrix.getrow(wordind).todense()

    def getSparseRow(self, wordind):
        return self.matrix.getrow(wordind)

    def printnonzeros(self, wordind):
        wrd_vec = np.squeeze(np.asarray(self.matrix.getrow(wordind).todense()))
        cnt = 0
        for i in wrd_vec:
            if not i == 0.0:
                print str(cnt) + ": " + str(i)
            cnt += 1

    def getrowsums(self, matrixdata):
        ret = []
        for i in matrixdata:
            ret.append(sum(i))
        return ret

    def ppmiTransform(self):
        '''
        transform matrix to  ppmi weighting as in:
        Turney, Peter D., and Patrick Pantel. 
        From frequency to meaning: Vector space models of semantics. 
        Journal of artificial intelligence research 37.1 (2010): 141-188.
        '''
        # compute the sum of each row and save it in rowSums matrix [number_of_rows, 1]
        rowSums = self.matrix.sum(1)
        # compute the sum of each column and save it in colSums matrix [1, number_of_columns]
        colSums = self.matrix.sum(0)
        # compute the sum of all values in the matrix by summing over all rowSums
        # we could also sum over colSums if we knew the number of columns is much less than rows
        totalSum = rowSums.sum()
        
        # row and col indexes of nonzero values
        rows, cols = self.matrix.nonzero()
        for row, col in zip(rows, cols):
            pmi = log(float(self.matrix[row, col] * totalSum) / (rowSums[row, 0] * colSums[0, col]))
            if pmi > 0.0:
                self.matrix[row, col] = pmi
            else:
                self.matrix[row, col] = 0.0

    def lmi(self):
        rowsums = []
        for i in self.matrix.data:
            rowsums.append(sum(i))
        colsums = []
        for i in self.matrix.transpose().data:
            colsums.append(sum(i))
        matsum = sum(rowsums) + sum(colsums)
        rows, cols = self.matrix.nonzero()
        for row, col in zip(rows, cols):
            pmi = math.log((self.matrix[row, col] * matsum) / (rowsums[row] * colsums[col]))
            if pmi > 0.0:
                self.matrix[row, col] = self.matrix[row, col] * pmi
            else:
                self.matrix[row, col] = 0.0

    def tfidf(self):
        self.matrix = TfidfTransformer().fit_transform(self.matrix)

    def t_svd(self, dimen):
        U, s, V = sparsesvd(sc.lil_matrix.tocsc(self.matrix), dimen)
        del s, V
        return U.T

    def t_pca(self, dimen):
        pca = sklearn.decomposition.PCA(dimen)
        pca.fit(sc.lil_matrix.todense(self.matrix))
        return pca.transform(sc.lil_matrix.todense(self.matrix))

    def tocsr(self):
        return sc.lil_matrix.tocsr(self.matrix)

    def delete(self):
        del self.matrix
