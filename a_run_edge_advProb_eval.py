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
    print "For S = %d"%(s)
    numerator[s] = A[s].dot(Ap.T)
    denominator[s] = A[s].dot(I.T)
numeratorP = Ap.dot(Ap.T)
denominatorP = Ap.dot(I.T)
print '\t%.2f seconds elapsed'%(time.clock()-start)

print 'Creating normalization for star=5...'
numerator[5].sort_indices()
denominator[5].sort_indices()
Ncoo5 = sp.coo_matrix(numerator[5])
Dcoo5 = sp.coo_matrix(denominator[5])
offset5 = np.ones((len(Dcoo5.data),))
Noff5 = sp.csr_matrix((offset5, (Dcoo5.row, Dcoo5.col)),shape=[B,B])
Doff5 = sp.csr_matrix(((offset5.dot(2)), (Dcoo5.row, Dcoo5.col)),shape=[B,B])
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
Psimple = {}
for busID in Simple :
    posRate = sum(1 for i in Simple[busID] if i in pos_list)
    Psimple[busID] = posRate / float(len(Simple[busID]))
print '\t%.2f seconds elapsed'%(time.clock()-start)

print 'Calculating advanced probability...'
Numerator5 = numerator[5] + Noff5
Denominator5 = denominator[5] + Doff5
Numerator5.sort_indices()
Denominator5.sort_indices()
Dcoo5 = sp.coo_matrix(Denominator5)
Pdat5 = Numerator5.tocoo().data / Dcoo5.data
P5 = sp.csr_matrix((Pdat5, (Dcoo5.row, Dcoo5.col)),shape=[B,B])

NumeratorP = numeratorP + NoffP
DenominatorP = denominatorP + DoffP
NumeratorP.sort_indices()
DenominatorP.sort_indices()
DcooP = sp.coo_matrix(DenominatorP)
PdatP = NumeratorP.tocoo().data / DcooP.data
Pp = sp.csr_matrix((PdatP, (DcooP.row, DcooP.col)),shape=[B,B])
print '\t%.2f seconds elapsed'%(time.clock()-start)

print 'Calculating combined probability...'
P5combo = {};
PPcombo = {};
for busID in Simple :
    P5combo[busID] = 0
    PPcombo[busID] = 0

    crossBus = Pp.getrow(busID).tocoo().col
    for x in crossBus :
        P5combo[busID] += (P5[busID,x]) - (Psimple[busID])
        PPcombo[busID] += (Pp[busID,x]) - (Psimple[busID]) 

    if len(crossBus) > 0 :
        P5combo[busID] = (Psimple[busID]) + (P5combo[busID]/len(crossBus))
        PPcombo[busID] = (Psimple[busID]) + (PPcombo[busID]/len(crossBus)) 
    else :
        P5combo[busID] = (Psimple[busID]) 
        PPcombo[busID] = (Psimple[busID])
#    print 'for %d ... P5 = %.2f%%\tPP = %.2f%%'%(busID,P5combo[busID]*100,PPcombo[busID]*100)
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
        bid_lst.append(b)
        label_lst.append(l)
        probability = PPcombo[b]

        if probability > 0.0001 :
                probability = float(math.log(probability))

        if b not in PPcombo :
            score_lst.append(0.0)
        else : 
            score_lst.append(probability)

        print 'Bid:%s  | S:%.6f'%(b, probability);
    fid.write('\n'.join(['%s %.6f %d'%(x[0],x[1],x[2]) for x in zip(bid_lst, score_lst, label_lst)])+'\n');
    fid.close();
print '   %.2f seconds elapsed'%(time.clock()-start);

