import json
import scipy.sparse as sp
import numpy as np
import pickle
import time
import sklearn.metrics as metrics
import util_functions as util

import sys, os
sys.path.append('/home/hltcoe/gsell/tools/python_mods/');
import plot_tools as plt

#sample run: python -i a_run_edge_advProb_eval.py posneg234out 1_5

edge_type = sys.argv[1];
test_cond = sys.argv[2];

start = time.clock();

execfile('load_edge_attr_data.py');
#to_save['Train Reviewer List'] = train_reviewer_lst;
#to_save['Test Reviewer List'] = test_reviewer_lst;
#to_save['Business Information'] = bid_info;
#to_save['Review Information'] = review_info;
#to_save['Reviewer Reviews'] = reviewer_reviews;
#to_save['Reviewed Business List'] = reviewed_bid_lst;

B = len(data['Reviewed Business List']);
R = len(data['Train Reviewer List']);

print 'Building index lookups...'
reviewer_idx = {};
for n in range(R) : reviewer_idx[data['Train Reviewer List'][n]] = n; #end
business_idx = {};
for n in range(B) : business_idx[data['Reviewed Business List'][n]] = n; #end
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Building Business x Reviewer index...'
r = {};
c = {};
A = {};
for s in [1,2,3,4,5] :
    r[s] = [];
    c[s] = [];

for uid in data['Train Reviewer List'] :
    for rid in data['Reviewer Reviews'][uid] :
        stars = int(float(data['Review Information'][rid]['stars']));
	# For each star rating, save index value of associated business_idx and reviewer_idx
        r[stars].append(business_idx[data['Review Information'][rid]['business_id']]);
        c[stars].append(reviewer_idx[uid]);
for s in r.keys() :
    # map the reviewer's reviews to businesses
    #  the np.ones creates a matrix of len(r[s]) of ones
    A[s] = sp.csr_matrix((np.ones((len(r[s]),)),(r[s],c[s])),shape=[B,R]);
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Building graph...'
if 'posneg' in edge_type :
    Ap = sp.csr_matrix(([],([],[])),shape=[B,R]);
    An = sp.csr_matrix(([],([],[])),shape=[B,R]);
    if '3neg' in edge_type :
        pos_list = [4,5];
        neg_list = [1,2,3];
    elif '3pos' in edge_type :
        pos_list = [3,4,5];
        neg_list = [1,2];
    elif '3out' in edge_type :
        pos_list = [4,5];
        neg_list = [1,2];
    elif '234out' in edge_type :
        pos_list = [5];
        neg_list = [1];
    else :
        print 'Unknown edge type '+edge_type;
        sys.exit();
    for s in pos_list : Ap += A[s]; #end
    for s in neg_list : An += A[s]; #end

numerator = sp.csr_matrix(([],([],[])),shape=[R,R]);
denominator = sp.csr_matrix(([],([],[])),shape=[R,R]);
print "Creating the transpose map for numerator";
for s in pos_list :
    print "For S = %d"%(s);
    numerator += sp.csr_matrix(A[s].T.dot(Ap));
    denominator += sp.csr_matrix(A[s].T.dot(A[s]));
print '   %.2f seconds elapsed'%(time.clock()-start);

print "You have the numerator now with: "
print "\tshape: " + str(numerator.shape);
print "\tstored elements: " + str(numerator.getnnz());
print "\n You have the denominator now with: "
print "\tshape: " + str(denominator.shape);
print "\tstored elements: " + str(denominator.getnnz());
print '   %.2f seconds elapsed'%(time.clock()-start);

print "Extracting RCD for numerator & denominator"
numcol = sp.coo_matrix(numerator)
demcol = sp.coo_matrix(denominator)
numcol_r = numcol.row
numcol_c = numcol.col
numcol_d = numcol.data
demcol_r = demcol.row
demcol_c = demcol.col
demcol_d = demcol.data
print '   %.2f seconds elapsed'%(time.clock()-start);
#halfeq = sp.linalg.inv(denom)
#lu = sp.linalg.splu(denom)
#eye = np.eye(ksize)
#halfeq = lu.solve(eye)
#k = numerator.dot(halfeq);
#k = sp.linalg.spsolve(denom, numerator)


#############################
# CURRENT KILL LINE
#############################

K = sp.coo_matrix(K);
s = K.sum(axis=0);
S = sp.spdiags(1/s,0,K.shape[0],K.shape[1]);
P = K.dot(S);
print '   %.2f seconds elapsed'%(time.clock()-start);




print 'Prepping evaluation...';
np.random.seed(159);
if max_iter > 0 :
    score_dir = 'projects/scores/single_%s_%03d_%04d_%s'%(edge_type,max_step,max_iter,test_cond);
    K = sp.coo_matrix(K);
    bid_r = K.row;
    bid_c = K.col;
    bid_d = K.data;
    edge_idx = {};
    edge_w = {};
    for n in range(K.shape[0]) : 
        edge_idx[n] = []; 
        edge_w[n] = [];    
    #end
    for n in range(len(bid_r)) :
        edge_idx[bid_r[n]].append(bid_c[n]);
        edge_w[bid_r[n]].append(bid_d[n]);
    #end
else :
    score_dir = 'projects/scores/single_%s_%03d_post_%s'%(edge_type,max_step,test_cond);
#end
os.system('mkdir -p %s'%(score_dir));
os.system('rm %s/*'%(score_dir));
print '   %.2f seconds elapsed'%(time.clock()-start);

if max_iter > 0 :
    print 'Running evaluation (%d steps with %d simulations)...'%(max_step,max_iter);
else :
    print 'Running evaluation (%d steps)...'%(max_step);
#end
for r in test_reviewer_lst :
    [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,r),business_idx);
    occupation = np.zeros((K.shape[0],));
    if max_iter > 0 :
        for (b,i,l) in train_lst :
            for n in range(max_iter) :
                if l == 1 :
                    bid = i;
                else :
                    if 'star' in edge_type :
                        bid = i+4*B;
                    else :
                        bid = i+B;
                    #end
                #end
                occupation[bid] += 1;
                for m in range(max_step) :
                    cs = np.cumsum(edge_w[bid]);
                    bid = edge_idx[bid][np.argmax(cs[-1]*np.random.rand()<cs)];
                    occupation[bid] += 1;
                #end
            #end
        #end
    else :
        o = np.zeros_like(occupation);
        for (b,i,l) in train_lst :
            if l == 1 :
                bid = i;
            else :
                if 'star' in edge_type :
                    bid = i+4*B;
                else :
                    bid = i+B;
                #end
            #end
            o[bid] += 1;
        #end
        occupation += o;
        for n in range(max_step) :
            o = P.dot(o);
            occupation += o;
        #end
    #end
    combo_o = np.zeros((B,));
    for n in range(len(recombo_weights)) :
        combo_o += recombo_weights[n]*occupation[recombo_bounds[n]:recombo_bounds[n+1]];
    #end
    err = util.evaluate_scores(combo_o,test_lst,outfile='%s/%s.scores'%(score_dir,r));
    print '   %.2f seconds elapsed'%(time.clock()-start);
#end

os.system('python score_rank_list.py -l lists_%s/ -s %s'%(test_cond,score_dir));
