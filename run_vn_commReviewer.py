# Script takes in A as saved in memory for data
# Creates two files:
#     * egivals listed in embedding_eigvals.txt
#     * context_features in embedding_conFeat.txt
#
# run as "python3 run_vn_commReviewer.py yelp_csr_matrix.npz"
#    where yelp_csr_matrix.npz is the saved data for 
#    csr_matrix A
import numpy as np
import scipy
import scipy.sparse as sp
import time
import sys, os
import sklearn.preprocessing as pp

sys.path.append('/home/hltcoe/apfannenstein/AttrVN')
from attr_vn import *

if len(sys.argv) < 2 :
    print('\nERROR: Missing arguments for load file')
    print('\tpython3 run_vn_commReviewer.py csv_load_file.npz')
    sys.exit()


# Print time elapse in seconds && minutes
start = time.clock()
def printTime() :
    timeElapse = time.clock()-start
    print('\t%.2f seconds elapsed -- %.2f minutes'%(timeElapse, timeElapse/60))

# Calculate cosine distance from a sparse matrix
def cosine_similiaries(mat) :
    col_normed = pp.normalize(mat.tocsc(), axis=0)
    return col_normed.T * col_normed


filename = sys.argv[1]
print("Loading %s from memory"%(filename))
loader = np.load(filename)
A = sp.csr_matrix((loader['data'], loader['indicies'], loader['indptr']), shape = loader['shape'])
printTime()

print("Creating business x business matrix")
#B = A.dot(A.T)
#B = cosine_similarities(B)
printTime()

print("running embed_symmetric_operator")
B = SymmetricSparseLinearOperator(symmetrize_sparse_matrix(B))
(eigvals, context_features) = embed_symmetric_operator(B, embedding = 'adj+diag', k = 1000, tol=None, verbose=False)
printTime()

print("Minimizing features")
eigvals = eigvals[:50]
context_features_test = np.matrix(context_features[:,:50])
with open('embedding_eigvals.txt', 'w') as fp:
    fp.write('\n'.join('%f' %val for val in eigvals))
with open('embedding_conFeat.txt', 'w') as fp:
    for line in context_features_test :
        np.savetxt(fp, line, fmt='%.4f')
printTime()

print("Computing X*S^(1/2)")
S = asmatrix(scipy.linalg.sqrtm(eigvals))**2
X = context_features_test
M = X.dot(S)
with open('vn_math.result', 'w') as fp:
    for line in M :
        np.savetxt(fp, line, fmt='%.4f')
printTime()
