# python2
# make_csr_yelp.py
# this script creates the sparse matrix for the Yelp data and saves
# ..to file for ingest into a python3 script

import json
import scipy.sparse as sp
import numpy as np
import pickle
import time
import util_functions as util
import sys, os
import math
import operator
from scipy.sparse.linalg import eigsh
import sklearn.preprocessing as pp


#sys.path.append('/home/hltcoe/apfannenstein/AttrVN')
#from attr_vn import *

sys.path.append('/home/hltcoe/gsell/tools/python_mods/')

review_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_review.json';
DEFAULT_IDF_UNITTEST = 1.0
test_cond = '12_45'
start = time.clock();
pos_lst = [4,5];
neg_lst = [1,2];

# Print time elapse in seconds && minutes
def printTime() :
    timeElapse = time.clock()-start
    print'\t%.2f seconds elapsed -- %.2f minutes'%(timeElapse, timeElapse/60)
printTime()

def cosine_similarities(mat) :
    col_normed_map = pp.normalize(mat.tocsc(), axis=0)
    return col_normed_map.T * col_normed_map


reviewer_idx = {}
business_idx = {}
review_idx = {}
#os.system('python load_edge_attr_data.py')
#exec(open('load_edge_attr_data.py').read())
execfile('load_edge_attr_data.py')
print '\nBuilding index lookups...'
B = len(data['Reviewed Business List'])
R = len(data['Train Reviewer List'])
for n in range(R) : reviewer_idx[data['Train Reviewer List'][n]] = n
for n in range(B) : business_idx[data['Reviewed Business List'][n]] = n


print'\nBuilding business review idx'
r = []  #reviewer 
c = []  #business 
for uid in reviewer_idx :
    for rid in data['Reviewer Reviews'][uid] :
        reviewInfo = data['Review Information'][rid]
        bid = reviewInfo['business_id']

        r.append(reviewer_idx[uid])     # r
        c.append(business_idx[bid])     # c
A = sp.csr_matrix((np.ones((len(c),)),(c,r)),shape=[B,R])
A = A.dot(A.T)
printTime()

print'\nDetermining Cosine Similarities'
sims = cosine_similarities(A)
printTime()

print'\nSaving CSR matrix to file'
filename = 'yelp_csr_cos_matrix.npz'
#sp.save_npz('yelp_csr_matrix.npz', A)
np.savez(filename, data = sims.data, indicies = sims.indices, indptr = sims.indptr, shape = sims.shape)
printTime()

