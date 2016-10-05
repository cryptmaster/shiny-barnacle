import json
import scipy.sparse as sp
import numpy as np
import pickle
import time
import sklearn.metrics as metrics
import util_functions as util
import sys, os, os.path
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
for n in range(len(data['Train Reviewer List'])) : reviewer_idx[data['Train Reviewer List'][n]] = n; #end
business_idx = {};
for n in range(len(data['Reviewed Business List'])) : business_idx[data['Reviewed Business List'][n]] = n; #end
print '   %.2f seconds elapsed'%(time.clock()-start);

# Create list of positive and negative reviews
# each list contains the BID from reviews for indexing
print '\nBuilding Business x Reviewer index...'
business_revs = {}; 
pos_bus_revs = {};
r_pos = [];
r_neg = [];
for uid in reviewer_idx :
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

        if stars in pos_list :
            r_pos.append(rid);
	    pos_bus_revs[bid].append(reviewID);
        elif stars in neg_list :
            r_neg.append(rid);
    #end
#end
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
print '   %.2f seconds elapsed'%(time.clock()-start);


print 'Running evaluation...';
score_dir = 'scores/probability_%s'%(test_cond);
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

    # For each reviewer, find the buses reviewed
    # and write the bid, prob/score, and a label to file
    # such that all buses rev'd by UID are in single file
    for value in test_lst :
        bid = value[0];
	bid_lst.append(bid);
        label_lst.append(value[2]);
	if bid not in bus_rank :
	    score_lst.append(0.0);
	else :
	    score_lst.append(bus_rank[bid]['prob']);
    #end
    fid.write('\n'.join(['%s %.6f %d'%(x[0],x[1],x[2]) for x in zip(bid_lst, score_lst, label_lst)])+'\n');
    fid.close();
#end
print '   %.2f seconds elapsed'%(time.clock()-start);

