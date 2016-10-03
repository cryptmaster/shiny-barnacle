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

pos_list = [int(float(x)) for x in sys.argv[1].split(',') if len(x)>0] #5 or 4,5
neg_list = [int(float(x)) for x in sys.argv[2].split(',') if len(x)>0] #1 or 1,2
test_cond = sys.argv[3]; #diff between test sets

start = time.clock();

execfile('load_edge_attr_data.py');
#to_save['Train Reviewer List'] = train_reviewer_lst;
#to_save['Test Reviewer List'] = test_reviewer_lst;
#to_save['Business Information'] = bid_info;
#to_save['Review Information'] = review_info;
#to_save['Reviewer Reviews'] = reviewer_reviews;
#to_save['Reviewed Business List'] = reviewed_bid_lst;

# lookup pair of reviewer/businesses linked to integer for indexing
print 'Building index lookups...'
reviewer_idx = {};
for n in range(len(data['Train Reviewer List'])) : 
    reviewer_idx[data['Train Reviewer List'][n]] = n; 
#end
business_idx = {};
for n in range(len(data['Reviewed Business List'])) : 
    business_idx[data['Reviewed Business List'][n]] = n; 
#end
print 'sizeof business_idx = %d' %(len(business_idx));
print '   %.2f seconds elapsed'%(time.clock()-start);

# Create list of positive and negative reviews
# each list contains the BID from reviews for indexing
print '\nBuilding Business x Reviewer index...'
business_revs = {}; 
pos_bus_revs = {};
reviewer_revs = {};
r_pos = [];
r_neg = [];
for uid in data['Train Reviewer List'] :
    for rid in data['Reviewer Reviews'][uid] :
	reviewInfo = data['Review Information'][rid];
        stars = float(reviewInfo['stars']);
	bid = reviewInfo['business_id'];
        reviewID = reviewInfo['review_id'];

	if bid not in business_revs :
   	    business_revs[bid] = {};
	    pos_bus_revs[bid] = [];
	if reviewID not in business_revs[bid] : 
	    business_revs[bid][reviewID] = [];
        business_revs[bid][reviewID].append(rid);

	review = [];
	review.append(bid);
	
        if stars in pos_list :
            r_pos.append(rid);
	    pos_bus_revs[bid].append(reviewID);
	    review.append(1);
        elif stars in neg_list :
            r_neg.append(rid);
	    review.append(-1);

	if rid in reviewer_idx :
	    if rid not in reviewer_revs :
	        reviewer_revs[rid] = [];
	    reviewer_revs[rid] = review;
    #end
#end
print '    postive: %d     negative: %d' %(len(r_pos),len(r_neg));
print '    number of businesses %d...' %(len(business_revs));
print '    %.2f seconds elapsed'%(time.clock()-start);

print '\nDetermining probability...';
bus_rank = {};
for bid in business_revs :
    probability = len(pos_bus_revs[bid]) / float(len(business_revs[bid]));
    rank = {'prob':probability, 'positive':len(pos_bus_revs[bid]), 'total':len(business_revs[bid])};
    bus_rank[bid] = rank; 
#end
print '   %.2f seconds elapsed'%(time.clock()-start);

print '\nSorting the Reviews...';
sortedList = sorted(bus_rank.iteritems(), key=lambda (x, y): (y['prob'], y['total']));
#for item in sorted(bus_rank.iteritems(), key=lambda (x, y): (y['prob'], y['total'])) :
#    print item;
#end
print '   %.2f seconds elapsed'%(time.clock()-start);

print '\nWriting to .scores files...';
for rid in reviewer_revs :
    outfile = rid.join('.scores');
    fid = open(outfile,'w');
    lst_ids = [];
    lst_scores = [];
    lst_labels = [];
    for item in rid :
	lst_ids.append(item[0]); 
	prob = bus_rank[bid]['prob'];
	lst_scores.append(prob);
	lst_labels.append(item[1]);
	fid.write('\n'.join(['%s %.6f %d'%(x[0],x[1],x[2]) for x in zip(lst_ids, lst_scores, lst_labels)])+'\n');
    #end
    fid.close();
#end
print '   %.2f seconds elapsed'%(time.clock()-start);


#print 'Running evaluation (%d steps)...'%(max_step);
#np.random.seed(159);
#score_dir = 'gsell_yelp/projects/scores/single_%s_posterior_%s'%(graph_type,test_cond);
#os.system('mkdir -p %s'%(score_dir));
#os.system('rm %s/*'%(score_dir));
#prior = np.zeros((len(test_reviewer_lst),));
#ap = np.zeros((len(test_reviewer_lst),));
#auc = np.zeros((len(test_reviewer_lst),));
#mean_rank_ratio = np.zeros((len(test_reviewer_lst),));
#median_rank_ratio = np.zeros((len(test_reviewer_lst),));
#eer = np.zeros((len(test_reviewer_lst),));
#rho = np.zeros((len(test_reviewer_lst),));
#p = np.zeros((len(test_reviewer_lst),));
#for r in test_reviewer_lst :
#    [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,r),business_idx);
#    occupation = np.zeros((len(data['Reviewed Business List']),));
#    o = np.zeros_like(occupation);
#    for (b,i,l) in train_lst :
#        o[i] += l;
#    #end

#    occupation += o;
#    for n in range(max_step) :
#        o = P.dot(o);
#        occupation += o;
#    #end

#    err = util.evaluate_scores(occupation,test_lst,outfile='%s/%s.scores'%(score_dir,r));
#    idx = test_reviewer_lst.index(r);
#    prior[idx] = err[0];
#    ap[idx] = err[1];
#    auc[idx] = err[2];
#    mean_rank_ratio[idx] = err[3];
#    median_rank_ratio[idx] = err[4];
#    eer[idx] = err[5];
#    rho[idx] = err[6];
#    p[idx] = err[7].data;
#    print 'RESULT: %s (%.4f) %.4f %.4f %.4f %.4f %.4f %.4f (%.4f)'%(r,prior[idx],ap[idx],auc[idx],mean_rank_ratio[idx],median_rank_ratio[idx],eer[idx],rho[idx],p[idx]);
#    print '   %.2f seconds elapsed'%(time.clock()-start);
##end

#print 'Occupation posteriors on single graph'
#print 'Graph Type: %s'%(graph_type);
#print 'Positive edges: %s'%(sys.argv[2]);
#print 'Negative edges: %s'%(sys.argv[3]);
#print 'Steps in walk: %d'%(max_step);
#print 'Total walks per training example: N/A';
#print 'AVERAGE: (%.4f) %.4f %.4f %.4f %.4f %.4f %.4f'%(np.mean(prior),np.mean(ap),np.mean(auc),np.mean(mean_rank_ratio),np.mean(median_rank_ratio),np.mean(eer),np.mean(rho));
#
#os.system('python score_rank_list.py -l lists_%s/ -s %s'%(test_cond,score_dir));
