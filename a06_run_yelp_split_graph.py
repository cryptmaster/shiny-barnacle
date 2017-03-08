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

edge_type = sys.argv[1];
max_step = int(sys.argv[2]);
max_iter = int(sys.argv[3]);
test_cond = sys.argv[4];

# Edge type options:  arg1_{count/binary}
# connectposneg3pos
# connectposneg3neg
# connectposneg3out
# connectposneg234out
# splitposneg3pos
# splitposneg3neg
# splitposneg3out
# splitposneg234out
# connectstar
# splitstar

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
#end
for uid in data['Train Reviewer List'] :
    for rid in data['Reviewer Reviews'][uid] :
        stars = int(float(data['Review Information'][rid]['stars']));
        r[stars].append(business_idx[data['Review Information'][rid]['business_id']]);
        c[stars].append(reviewer_idx[uid]);
    #end
#end
for s in r.keys() :
    A[s] = sp.csr_matrix((np.ones((len(r[s]),)),(r[s],c[s])),shape=[B,R]);
#end
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
    #end
    for s in pos_list : Ap += A[s]; #end
    for s in neg_list : An += A[s]; #end
    if 'connect' in edge_type :
        d_t = np.append(Ap.data,An.data);
        ind_t = np.append(Ap.indices,An.indices);
        idp_t = np.append(Ap.indptr,An.indptr[1:]+len(Ap.indices));
        At = sp.csr_matrix((d_t,ind_t,idp_t),shape=[2*B,R]);
        K = At.dot(At.T);
    elif 'split' in edge_type :
        Kp = Ap.dot(Ap.T);
        Kn = An.dot(An.T);
        d_t = np.append(Kp.data,Kn.data);
        ind_t = np.append(Kp.indices,Kn.indices+B);
        idp_t = np.append(Kp.indptr,Kn.indptr[1:]+Kp.indptr[-1]);
        K = sp.csr_matrix((d_t,ind_t,idp_t),shape=[2*B,2*B]);
    #end
    recombo_bounds = [0,B,2*B];
    recombo_weights = [1,-1];
elif 'star' in edge_type :
    d_t = np.array([]);
    ind_t = np.array([]);
    idp_t = np.array([0]);
    if 'connect' in edge_type :
        offset=0;
        for s in [1,2,3,4,5] : 
            Ks = A[s].dot(A[s].T);
            d_t = np.append(d_t,Ks.data);
            ind_t = np.append(ind_t,Ks.indices);
            idp_t = np.append(idp_t,Ks.indptr[1:]+offset);
            offset = idp_t[-1];
        #end
        K = sp.csr_matrix((d_t,ind_t,idp_t),shape=[5*B,5*B]);
    elif 'split' in edge_type :
        offset=0;
        for s in [5,4,3,2,1] : 
            d_t = np.append(d_t,A[s].data);
            ind_t = np.append(ind_t,A[s].indices+(s-1)*B);
            idp_t = np.append(idp_t,A[s].indptr[1:]+offset);
            offset = idp_t[-1];
        #end
        At = sp.csr_matrix((d_t,ind_t,idp_t),shape=[5*B,R]);
        K = At.dot(At.T);
    #end
    recombo_bounds = [0,B,2*B,3*B,4*B,5*B,6*B];
    recombo_weights = [5,4,3,2,1];
#end
if 'binary' in edge_type : K[K>0] = 1; #end
#K = sp.coo_matrix(K);
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
