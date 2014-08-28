'''
Created on Dec 11, 2013

@author: af
'''
from math import sqrt
from scipy.sparse import csr_matrix
import unittest


class MatrixOperationTest(unittest.TestCase):


    def testMultiply(self):
        print "Multiply Test"
        rows = [0, 0, 0]
        columns = [0, 4, 7]
        data = [+1, +2, -1]
        m1 = csr_matrix((data, (rows, columns)), shape=(1, 10))
        print "m1: ", m1.todense()
        
        rows2 = [0, 0, 0, 0]
        columns2 = [0, 4, 7, 2]
        data2 = [+1, +2, -5, 7]
        m2 = csr_matrix((data2, (rows2, columns2)), shape=(1, 10))
        print "m2: ", m2.todense()
        
        print "m1.multiply(m2): ", m1.multiply(m2).todense()
        
        print "m1.multiply(m2).sum(): ", m1.multiply(m2).sum()
        
    def testDot(self):
        print "Dot product Test"
        rows = [0, 0, 0]
        columns = [0, 4, 7]
        data = [+1, +2, -1]
        m1 = csr_matrix((data, (rows, columns)), shape=(1, 10))
        print "m1: ", m1.todense()
        
        rows2 = [0, 0, 0, 0]
        columns2 = [0, 4, 7, 2]
        data2 = [+1, +2, -5, 7]
        m2 = csr_matrix((data2, (rows2, columns2)), shape=(1, 10))
        print "m2: ", m2.todense()
        print "m2.transpose(): ", m2.transpose().todense()
        print "m1.dot(m2.transpose()): ", m1.dot(m2.transpose()).todense()
        
        print "m1.dot(m2.transpose()).sum(): ", m1.dot(m2.transpose()).sum()
    
    def testCosineSimilarity(self):
        print "Cosine Similarity Test"
        rows = [0, 0, 0]
        columns = [0, 4, 7]
        data = [+0.1, +0.2, -0.1]
        m1 = csr_matrix((data, (rows, columns)), shape=(1, 10))
        print "m1: ", m1.todense()
        
        rows2 = [0, 0, 0]
        columns2 = [0, 4, 7]
        data2 = [+0.1, +0.2, -0.1]
        m2 = csr_matrix((data2, (rows2, columns2)), shape=(1, 10))
        print "m2: ", m2.todense()
        
        print "m1.multiply(m2).sum(): ", m1.multiply(m2).sum()
        
        print "vector size m1: ", sqrt(m1.multiply(m1).sum())
        print "vector size m2: ", sqrt(m2.multiply(m2).sum())
        
        print "cosine similarity: m1.m2 / |m1|*|m2| : ", m1.multiply(m2).sum() / (sqrt(m1.multiply(m1).sum() * m2.multiply(m2).sum()))

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
