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

if len(sys.argv) < 5 :
    print '\n\nMissing args at execution.'
    print 'Sample execution (where optional items in []: '
    print '\tpython [-i] a_run_edge_advProb_eval.py edge_type test_cond max_iter max_step\n\n'
    sys.exit()

edge_type = sys.argv[1];
test_cond = sys.argv[2];
max_iter = sys.argv[3]
max_step = sys.argv[4]
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
b = {};
for s in [1,2,3,4,5] :
    r[s] = [];
    c[s] = [];
for uid in data['Train Reviewer List'] :
    for rid in data['Reviewer Reviews'][uid] :
        stars = int(float(data['Review Information'][rid]['stars']));
	# For each star rating, save index value of associated business_idx and reviewer_idx
        busID = business_idx[data['Review Information'][rid]['business_id']];
        if busID not in b :
            b[busID] = {}
            b[busID]['rate'] = stars
            b[busID]['numRate'] = 1
            b[busID]['avg'] = float(stars/5.0)
        else :
            b[busID]['rate'] += stars
            b[busID]['numRate'] += 1
            b[busID]['avg'] = float(b[busID]['rate']) / (b[busID]['numRate'] * 5)
        r[stars].append(business_idx[data['Review Information'][rid]['business_id']]);
        c[stars].append(reviewer_idx[uid]);
I = sp.csr_matrix(([],([],[])),shape=[B,R]);
for s in r.keys() :
    # map the reviewer's reviews to businesses
    A[s] = sp.csr_matrix((np.ones((len(r[s]),)),(r[s],c[s])),shape=[B,R]);
    I += A[s];
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
print '\t%.2f seconds elapsed'%(time.clock()-start);

print 'Building Numerator && Denominator...'
numerator = {}
denominator = {}
for s in pos_list :
    print "For S = %d"%(s)
    numerator[s] = A[s].dot(Ap.T)
    denominator[s] = A[s].dot(I.T)
numeratorP = Ap.dot(Ap.T)
denominatorP = Ap.dot(I.T)
print '\t%.2f seconds elapsed'%(time.clock()-start)

print "You have the numerator now with: "
print "\tshape: " + str(numerator[5].shape) + "\tstored elements: " + str(numerator[5].getnnz())
print "You have the denominator now with: "
print "\tshape: " + str(denominator[5].shape) + "\tstored elements: " + str(denominator[5].getnnz())
print '\n\t%.2f seconds elapsed'%(time.clock()-start)

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

print 'Here\'s your current probabilities...'
#for x in range(0,B) :
#    busX = I[:,x]
#    tmpRow = Pp.getrow(x).tocoo().col
#    for y in tmpRow :
#        prob = P5[x,y] * 100
#        alProb = Pp[x,y] * 100
#        if prob != alProb :
#            busY = I[:,y]
#            numD5 = Numerator5[x,y]
#            denD5 = Denominator5[x,y]
#            numDP = NumeratorP[x,y]
#            denDP = DenominatorP[x,y]
#            bXav = float(b[x]['avg'])*100
#            bYav = float(b[y]['avg'])*100
#            bXn = b[x]['numRate']
#            bYn = b[y]['numRate']
#            print "(%d,%d)\tN5:%.0f D5:%.0f\tNP:%.0f DP:%.0f\tXav:%.2f%%=%d Yav:%.2f%%=%d\tP:%.2f%%, %.2f%%"%(x,y,numD5,denD5,numDP,denDP,bXav,bXn,bYav,bYn,prob,alProb)
#    print '*** %d COMPLETE, \t%.2f seconds elapsed'%(x,time.clock()-start)
    
#############################
# CURRENT KILL LINE
#############################

K = sp.coo_matrix(P5);
s = K.sum(axis=0);
S = sp.spdiags(1/s,0,K.shape[0],K.shape[1]);
P = K.dot(S);
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Prepping evaluation...';
np.random.seed(159);
if max_iter > 0 :
    score_dir = 'scores/single_%s_%03d_%04d_%s'%(edge_type,max_step,max_iter,test_cond);
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
    score_dir = 'scores/single_%s_%03d_post_%s'%(edge_type,max_step,test_cond);
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
