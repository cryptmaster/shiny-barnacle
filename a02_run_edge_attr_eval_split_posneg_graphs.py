import json
import scipy.sparse as sp
import numpy as np
import pickle
import time
import util_functions as util

import sys, os
sys.path.append('/home/hltcoe/gsell/tools/python_mods/');
import plot_tools as plt

graph_type = sys.argv[1];
pos_list = [int(float(x)) for x in sys.argv[2].split(',') if len(x)>0]
neg_list = [int(float(x)) for x in sys.argv[3].split(',') if len(x)>0]
max_step = int(sys.argv[4]);
max_iter = int(sys.argv[5]);

start = time.clock();

execfile('load_edge_attr_data.py');
#to_save['Train Reviewer List'] = train_reviewer_lst;
#to_save['Test Reviewer List'] = test_reviewer_lst;
#to_save['Business Information'] = bid_info;
#to_save['Review Information'] = review_info;
#to_save['Reviewer Reviews'] = reviewer_reviews;
#to_save['Reviewed Business List'] = reviewed_bid_lst;

print 'Building index lookups...'
reviewer_idx = {};
for n in range(len(data['Train Reviewer List'])) : reviewer_idx[data['Train Reviewer List'][n]] = n; #end
business_idx = {};
for n in range(len(data['Reviewed Business List'])) : business_idx[data['Reviewed Business List'][n]] = n; #end

print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Building Business x Reviewer index...'
r_pos = [];
c_pos = [];
r_neg = [];
c_neg = [];
for uid in data['Train Reviewer List'] :
    for rid in data['Reviewer Reviews'][uid] :
        stars = float(data['Review Information'][rid]['stars']);
        if stars in pos_list :
            r_pos.append(business_idx[data['Review Information'][rid]['business_id']]);
            c_pos.append(reviewer_idx[uid]);
        elif stars in neg_list :
            r_neg.append(business_idx[data['Review Information'][rid]['business_id']]);
            c_neg.append(reviewer_idx[uid]);
        #end
    #end
#end

A_pos = sp.csr_matrix((np.ones((len(r_pos),)),(r_pos,c_pos)),shape=[len(data['Reviewed Business List']),len(data['Train Reviewer List'])]);
A_neg = sp.csr_matrix((np.ones((len(r_neg),)),(r_neg,c_neg)),shape=[len(data['Reviewed Business List']),len(data['Train Reviewer List'])]);
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Building graph from common reviews...'
Kpp = A_pos.dot(A_pos.T);
Kpn = A_pos.dot(A_neg.T);
Knp = Kpn.T;
Knn = A_neg.dot(A_neg.T);

if graph_type == 'common_star_count' :
    Kp = Kpp;
    Kn = Knn;
elif graph_type == 'common_star_binary' :
    Kp = Kpp;
    Kn = Knn;
    Kp[Kp>0] = 1;
    Kn[Kn>0] = 1;
elif graph_type == 'common_star_percent' :
    Kp = Kpp;
    Kn = Knn;
    Kp[Kp>0] = Kp[Kp>0]/(Kpp[Kp>0]+Knn[Kp>0]+Kpn[Kp>0]+Knp[Kp>0]);
    Kn[Kn>0] = Kn[Kn>0]/(Kpp[Kn>0]+Knn[Kn>0]+Kpn[Kn>0]+Knp[Kn>0]);
#end
Kp = sp.coo_matrix(Kp);
bid_p_r = Kp.row;
bid_p_c = Kp.col;
bid_p_d = Kp.data;
p_edge_idx = {};
p_edge_w = {};
for n in range(Kp.shape[0]) : 
    p_edge_idx[n] = []; 
    p_edge_w[n] = [];    
#end
for n in range(len(bid_p_r)) :
    p_edge_idx[bid_p_r[n]].append(bid_p_c[n]);
    p_edge_w[bid_p_r[n]].append(bid_p_d[n]);
#end
Kn = sp.coo_matrix(Kn);
bid_n_r = Kp.row;
bid_n_c = Kp.col;
bid_n_d = Kp.data;
n_edge_idx = {};
n_edge_w = {};
for n in range(Kn.shape[0]) : 
    n_edge_idx[n] = []; 
    n_edge_w[n] = [];    
#end
for n in range(len(bid_n_r)) :
    n_edge_idx[bid_n_r[n]].append(bid_n_c[n]);
    n_edge_w[bid_n_r[n]].append(bid_n_d[n]);
#end
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Running evaluation (%d iterations with %d steps)...'%(max_iter,max_step);
np.random.seed(159);
os.system('mkdir -p projects/scores/posneg_%s'%(graph_type));
os.system('rm projects/scores/posneg_%s/*'%(graph_type));
prior = np.zeros((len(test_reviewer_lst),));
ap = np.zeros((len(test_reviewer_lst),));
auc = np.zeros((len(test_reviewer_lst),));
mean_rank_ratio = np.zeros((len(test_reviewer_lst),));
median_rank_ratio = np.zeros((len(test_reviewer_lst),));
eer = np.zeros((len(test_reviewer_lst),));
for r in test_reviewer_lst :
    [train_lst,test_lst] = util.read_key('lists/%s.key'%(r),business_idx);
    occupation = np.zeros((len(data['Reviewed Business List']),));
    for (b,i,l) in train_lst :
        if l == 1 :
            ei = p_edge_idx;
            ew = p_edge_w;
        else :
            ei = n_edge_idx;
            ew = n_edge_w;
        #end
        cnt_iter = 0;
        while cnt_iter < max_iter :
            bid = i;
            cnt_step = 0;
            occupation[bid] += l;
            while cnt_step < max_step :
                if len(ew[bid]) == 0 :
                    cnt_step = max_step;
                else :
                    cs = np.cumsum(ew[bid]);
                    bid = ei[bid][np.argmax(int(cs[-1]*np.random.rand())<cs)];
                    occupation[bid] += l;
                    cnt_step+=1;
                #end
            #end
            cnt_iter+=1;
        #end
    #end
    err = util.evaluate_scores(occupation,test_lst,outfile='projects/scores/posneg_%s/%s.scores'%(graph_type,r));
    idx = test_reviewer_lst.index(r);
    prior[idx] = err[0];
    ap[idx] = err[1];
    auc[idx] = err[2];
    mean_rank_ratio[idx] = err[3];
    median_rank_ratio[idx] = err[4];
    eer[idx] = err[5];
    print 'RESULT: %s (%.4f) %.4f %.4f %.4f %.4f %.4f'%(r,prior[idx],ap[idx],auc[idx],mean_rank_ratio[idx],median_rank_ratio[idx],eer[idx]);
    print '   %.2f seconds elapsed'%(time.clock()-start);
#end
print 'Random walks on split positive/negative graphs'
print 'Graph Type: %s'%(graph_type);
print 'Positive edges: %s'%(sys.argv[2]);
print 'Negative edges: %s'%(sys.argv[3]);
print 'Steps in walk: %d'%(max_step);
print 'Total walks per training example: %d'%(max_iter);
print 'OVERALL: (%.4f) %.4f %.4f %.4f %.4f %.4f'%(np.mean(prior),np.mean(ap),np.mean(auc),np.mean(mean_rank_ratio),np.mean(median_rank_ratio),np.mean(eer));
