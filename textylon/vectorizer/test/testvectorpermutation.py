'''
Created on Dec 10, 2013

@author: afshin rahimi
'''
from scipy.sparse import csr_matrix


rows = [0, 0, 0]
columns = [7, 4, 0]
data = [+1, +1, -1]
m = csr_matrix((data, (rows, columns)), shape=(1, 10))

indices = m.indices
print m.todense()
# right permutation
# m.indices = (indices + 1) % m.shape[1]

# left permutation
m.indices = (indices - 1) % m.shape[1]

print m.todense()
