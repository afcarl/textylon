'''
Created on Dec 16, 2013

@author: af
'''
import logging
from numpy import isinf
from numpy import isnan
from numpy import log
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cosine 
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sparsesvd import sparsesvd


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

class Matrix(object):
    '''
    This class is a wrapper for scipy sparse matrices. It uses
    CSR matrices now but it won't have any effect on the functionality \
    if the type of matrix is changed.
    '''
    matrix = None

    def __init__(self, dimensionX, dimensionY):
        '''
        Constructor
        '''
        self.matrix = lil_matrix((dimensionX, dimensionY))
    
    def getRow(self, rowID):
        return self.matrix.getrow(rowID).todense()
    
    def getSparseRow(self, rowID):
        return self.matrix.getrow(rowID)
    
    def getColumn(self, columnID):
        return self.matrix.getcol(columnID).todense()
    
    def getRowSum(self, rowID):
        return self.matrix.getrow(rowID).sum()
    
    def getColumnSum(self, columnID):
        return self.matrix.getcol(columnID).sum()
    
    def incrementValue(self, rowId, columnId, incrementValue):
        self.matrix[rowId, columnId] += incrementValue 
    
    def getCosineSimilarityByRowId(self, rowId1, rowId2):
        '''
        given two rowIds it returns the cosine similarity of the two
        corresponding vectors in the matrix. If the size of each is zero
        then it will return false.
        '''
        row1 = self.getRow(rowId1)
        row2 = self.getRow(rowId2)
        similarity = 1 - cosine(row1, row2)
        if isnan(similarity) or isinf(similarity):
            return False
        else:
            return similarity
    
    def getCosineSimilarityByRow(self, row1, row2):
        similarity = 1 - cosine(row1, row2)
        if isnan(similarity) or isinf(similarity):
            return False
        else:
            return similarity
    
    def tfidfTransform(self):
        '''
        The TFIDF transform will be applied on the matrix.
        '''
        logger.info("TFIDF transformation started.")
        TfidfTransformer().fit_transform(self.matrix)
        logger.info("TFIDF transformation finished successfully.")
    
    def svdTransform(self, dimension):
        '''
        This method uses SVD for dimension reduction given the dimensionality 
        of the produced matrix. The reduced matrix will be returned back.
        '''
        logger.info("SVD started with dimension = " + str(dimension))
        U, S, V = sparsesvd(self.matrix, dimension)
        logger.info("SVD finished successfully.")
        del S, V
        return U.T
    
    def pcaTransform(self, dimension):
        '''
        This method uses PCA dimension reduction and returns the reduced matrix
        '''
        logger.info("PCA started with dimension = " + str(dimension))
        pca = PCA(dimension)
        reducedMatrix = pca.fit_transform(self.matrix.todense())
        logger.info("PCA finished successfully.")
        return reducedMatrix
    
    def ppmiTransform(self):
        '''
        transform matrix to  ppmi weighting as in:
        Turney, Peter D., and Patrick Pantel. 
        From frequency to meaning: Vector space models of semantics. 
        Journal of artificial intelligence research 37.1 (2010): 141-188.
        '''
        logger.info("PPMI transform started...") 
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
        logger.info("PPMI transform finished successfully.")
    
    def lmiTransform(self):
        '''
        transform matrix to LMI weighting as in:
        Evert, Stefan. The statistics of word cooccurrences. 
        Diss. Dissertation, Stuttgart University, 2005.
        LMI is used to counteract the tendency of PMI to favor rare events.
        '''
        logger.info("LMI transform started...") 
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
                self.matrix[row, col] = pmi * self.matrix[row, col]
            else:
                self.matrix[row, col] = 0.0
        logger.info("LMI transform finished successfully.")
    
    def delete(self):
        del self.matrix
    
    
        
    
        
