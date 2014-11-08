'''
Created on 1 Sep 2014

@author: af
'''
import codecs
from scipy.sparse import lil_matrix, csr_matrix
import pickle
from sparsesvd import sparsesvd
import numpy as np
from os import path
from sklearn.metrics import mean_squared_error
from scipy.sparse import diags

HOME_DIR = '/home/arahimi/'
def csv2sparse(address, outputFile, separator=',', encoding='utf-8'):
    """Given the address of a [comma] separated file (rowName, columnName, value) 
    converts it to a sparse matrix.
    rowName and columnName are string values and value is double.
    """
    columns = {}
    rows = {}
    print "building the dictionaries..."
    with codecs.open(address, 'r', encoding) as inf:
        for line in inf:
            items = line.split(separator)
            if len(items) != 3:
                print "format error: the number of items is not 3."
                return -1
            rowName = items[0]
            columnName = items[1]
            # value = items[2]
            # add column and row names to dictionaries and find their matrix cell
            rowId = -1
            columnId = -1
            if rowName in rows:
                rowId = rows[rowName]
            else:
                rowId = len(rows)
                rows[rowName] = rowId
            
            if columnName in columns:
                columnId = columns[columnName]
            else:
                columnId = len(columns)
                columns[columnName] = columnId
    
    mat = lil_matrix((len(rows), len(columns)), dtype=None)
    print "matrix dimensions is " + str(mat.shape)
    print "building the matrix..."
    with codecs.open(address, 'r', encoding) as inf:
        for line in inf:
            items = line.split(separator)
            rowId = rows[items[0]]
            columnId = columns[items[1]]
            value = int(items[2])
            mat[rowId, columnId] = value
    
    
    # save the matrix
    print "saving the matrix in " + outputFile
    # with codecs.open(outputFile, 'wb') as outf:
        # pickle.dump((mat, rows, columns), outf)
    return mat

def readPickle(address):
    """
    read a pickle file and return a tuple (lil_matrix, rows_dictionary, columns_dictionary).
    rows_dictionary's keys are row names and its values are rowIds.
    columns_dictionary's keys are columns names and its values are columnIds.
    """
    with codecs.open(address, 'rb') as inf:
        mat, rows, columns = pickle.load(inf)
        return mat, rows, columns 
 
def factorize(orig_matrix, dimension):
    """
    given a matrix and a dimension, this function applies SVD and then
    returns the matrix resulted from the lower dimension space
    """
    print "SVD running with " + str(dimension) + " factor..."
    ut, s, vt = sparsesvd(orig_matrix, dimension)
    print "pickling the matrices..."
    # with codecs.open(path.join(HOME_DIR, 'svd'+ '_' +str(dimension).strip() + '.pickle'), 'wb') as outf:
    #    pickle.dump((ut, s, vt), outf)
    ut = csr_matrix(ut)
    s = csr_matrix(np.diag(s))
    vt = csr_matrix(vt)
    result = ut.T.dot(s.dot(vt))
    # result = np.dot(ut.T, np.dot(np.diag(s), vt))
    return result
     
mat = csv2sparse(path.join(HOME_DIR, 'stackdb.txt'), path.join(HOME_DIR, 'stackMatrix.pickle'))
# mat, rows, columns = readPickle(path.join(HOME_DIR, 'stackMatrix.pickle'))
dimension = 100
new_matrix = factorize(mat.tocsc(), dimension)
# with codecs.open(path.join(HOME_DIR, 'new_matrix'+ str(dimension).strip()+'.pickle')) as inf:
    # pickle.dump(new_matrix, inf)

print mean_squared_error(mat, new_matrix)
