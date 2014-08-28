from scipy import *
from scipy.sparse import *
from scipy.sparse import vstack


a = 'salam hale shoma'
b = ['b', 'b', 'c']
with open('/home/af/asdf.txt', 'w') as f:
    f.write(' '.join(a) + '\n')
    f.write(' '.join(b) + '\n')
row = array([0, 2, 2, 0, 1, 2, 2])
col = array([0, 0, 1, 2, 2, 2, 3])
data = array([1, 2, 3, 1, 5, 1, 7])
matrix = csc_matrix((data, (row, col)), shape=(3, 4))
print matrix.todense()

row2 = array([0, 0, 0])
col2 = array([0, 1, 2])
data2 = array ([1, 2, 3])
matrix2 = csc_matrix((data2, (row2, col2)), shape=(1, 3))
print "matrix 2"
print matrix2.todense()
row3 = array([0, 0, 0])
col3 = array([0, 1, 2])
data3 = array ([0, 0, 3])
matrix3 = csc_matrix((data3, (row3, col3)), shape=(1, 3))
print "matrix3"
print matrix3.todense()

print (matrix + matrix).todense()

print "sum"
print (matrix2 + matrix3).todense()

matrix2 = vstack(matrix2, matrix3)
print matrix2

rowsums = matrix.sum(1)
colsums = matrix.sum(0)

# print matrix.sum()
print rowsums
for i in range(0, matrix.shape[0]):
    print rowsums[i, 0]
print colsums
for i in range(0, matrix.shape[1]):
    print colsums[0, i]




matsum = rowsums.sum()
rows, cols = matrix.nonzero()
for row, col in zip(rows, cols):
    value = float(matrix[row, col] * matsum) / (rowsums[row, 0] * colsums[0, col])
    pmi = math.log(value)
    if pmi > 0.0:
        matrix[row, col] = pmi
    else:
        matrix[row, col] = 0.0

print matrix.todense()

