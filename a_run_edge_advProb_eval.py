import json
import scipy.sparse as sp
import numpy as np
import pickle
import time
import sklearn.metrics as metrics
import util_functions as util
import sys, os
import math
sys.path.append('/home/hltcoe/gsell/tools/python_mods/');
import plot_tools as plt

if len(sys.argv) < 3 :
    print '\n\nMissing args at execution.'
    print 'Execution (where suggested items in []) :'
    print '\tpython a_run_edge_advProb_eval.py [edge_type=posneg[3neg|3pos|3out|234out]] [test_cond=1_5|12_45]'
    sys.exit()

edge_type = sys.argv[1];
test_cond = sys.argv[2];
start = time.clock();

if 'posneg' in edge_type :
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
business_idx = {};
for n in range(R) : reviewer_idx[data['Train Reviewer List'][n]] = n; #end
for n in range(B) : business_idx[data['Reviewed Business List'][n]] = n; #end
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Building Business x Reviewer index...'
r = {};
c = {};
A = {};
Simple = {};
I = sp.csr_matrix(([],([],[])),shape=[B,R]);
for s in [1,2,3,4,5] :
    r[s] = [];
    c[s] = [];
for uid in data['Train Reviewer List'] :
    for rid in data['Reviewer Reviews'][uid] :
        reviewInfo = data['Review Information'][rid];
        stars = float(reviewInfo['stars']);
        busID = business_idx[reviewInfo['business_id']];

        if busID not in Simple :
            Simple[busID] = [];

        r[stars].append(busID);
        c[stars].append(reviewer_idx[uid]);
        Simple[busID].append(stars);
for s in r.keys() :
    A[s] = sp.csr_matrix((np.ones((len(r[s]),)),(r[s],c[s])),shape=[B,R]);
    I += A[s];
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Building graph...'
Ap = sp.csr_matrix(([],([],[])),shape=[B,R]);
numerator = {}
denominator = {}
for s in pos_list : Ap += A[s]; 
for s in pos_list :
    numerator[s] = A[s].dot(Ap.T)
    denominator[s] = A[s].dot(I.T)
numeratorP = Ap.dot(Ap.T)
denominatorP = Ap.dot(I.T)
print '\t%.2f seconds elapsed'%(time.clock()-start)


print 'Creating normalization for star=positive...'
numeratorP.sort_indices()
denominatorP.sort_indices()
NcooP = sp.coo_matrix(numeratorP)
DcooP = sp.coo_matrix(denominatorP)
offsetP = np.ones((len(DcooP.data),))
NoffP = sp.csr_matrix((offsetP, (DcooP.row, DcooP.col)),shape=[B,B])
DoffP = sp.csr_matrix(((offsetP.dot(2)), (DcooP.row, DcooP.col)),shape=[B,B])
print '\t%.2f seconds elapsed'%(time.clock()-start)

print 'Calculating simple probability...'
simd = []
for busID in Simple :
    posRate = sum(1 for i in Simple[busID] if i in pos_list)
    simd.append(posRate / float(len(Simple[busID]))) 
Psimple = sp.csc_matrix(simd).T
Psimple.sort_indices()
print '\t%.2f seconds elapsed'%(time.clock()-start)

print 'Calculating advanced probability...'
NumeratorP = numeratorP + NoffP
DenominatorP = denominatorP + DoffP
NumeratorP.sort_indices()  
DenominatorP.sort_indices()  
DcooP = sp.coo_matrix(DenominatorP)
PdatP = NumeratorP.tocoo().data / DcooP.data
Pp = sp.csr_matrix((PdatP, (DcooP.row, DcooP.col)),shape=[B,B])
print '\t%.2f seconds elapsed'%(time.clock()-start)


print 'Calculating combined probability...'
Ppcombo = Pp
Ppcombo.eliminate_zeros()
Psim_row = Psimple.T
Psim_row.eliminate_zeros()
# Take log of 'simple' prob & subtract that from 'adv' prob
#Psim_row.data = np.log(Psim_row.data)
#Ppcombo.data -= np.repeat(Psim_row.toarray()[0],np.diff(Ppcombo.indptr))
# Summation of each col in sparse matrix
# Add lg('simple' prob) + Sum('adv' prob); save as array
#P = Psim_row.toarray()[0] + np.squeeze(np.array(Ppcombo.sum(1)/(Ppcombo != 0).sum(1))) 
print '\t%.2f seconds elapsed'%(time.clock()-start)


#############################
## Enter data to .scores files
#############################
print 'Running evaluation...';
score_dir = 'scores/AdvancedProbability_%s_%s'%(edge_type,test_cond);
os.system('mkdir -p %s'%(score_dir));
os.system('rm %s/*'%(score_dir));
here = os.path.dirname(os.path.realpath(__file__));
for reviewer in test_reviewer_lst :
    [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,reviewer),business_idx);
    outfile = '%s/%s.scores'%(score_dir,reviewer);
    fid = open(outfile,'w');
    bid_lst = [];
    score_lst = [];
    label_lst = [];

    for (b,i,l) in test_lst :
        raterAvg = 0.0
        bsum = 0
        ctpos = 0
        # Find what rating given to each train from test
        for (b2,i2,l2) in train_lst :
            if l2 == 1:
                lookup = Ppcombo[i,i2]
                if lookup > 0 :
                    ctpos += 1
                    bsum += np.log(lookup) - np.log(Psim_row[0,i])
                    #print '\t\ttrain_lst[%s]: %d\tlookup:%.2f\tPsim:%.2f\tbsum:%.4f'%(b2,i2,lookup,Psim_row[0,i],bsum)
        if ctpos > 0:
            raterAvg = bsum/ctpos

        bid_lst.append(b)
        label_lst.append(l)
        probability = np.log(Psim_row[0,i]) + raterAvg
        # score protection
        if (float('-inf') < float(probability) < float('inf')) :
            score_lst.append(probability)  
        else:
            score_lst.append(0.0)  
        print 'Reviewer:%s\tBusiness:%s\t| Rate:%d\t| Bsum:%.2f\t| Score:%s'%(reviewer,b,l,bsum,str(probability));
	print '\tPsim_row[0,%s]: %.2f\t\t|ctpos: %d\t|bsum: %.2f\t|raterAvg=%.2f'%(i,Psim_row[0,i],ctpos,bsum,raterAvg)

    fid.write('\n'.join(['%s %.6f %d'%(x[0],x[1],x[2]) for x in zip(bid_lst, score_lst, label_lst)])+'\n');
    fid.close();
    print '\n'
    print '   %.2f seconds elapsed'%(time.clock()-start);
print '   %.2f seconds elapsed'%(time.clock()-start);

print'python score_rank_list.py -l lists_%s/ -s %s'%(test_cond,score_dir);
os.system('python score_rank_list.py -l lists_%s/ -s %s'%(test_cond,score_dir));
print '   %.2f seconds elapsed'%(time.clock()-start);

