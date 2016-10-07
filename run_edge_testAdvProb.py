# This is a copy of original run_edge_advProb_eval.py 
#  used to test modifications for speed
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
test_cond = sys.argv[3]; #diff between test sets.... 1_5

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
print '\nCompiling Review Maps... ';
businessRevs = {};
LbidMap = {};
TbidMap = {};
for user in reviewer_idx :
    for review in data['Reviewer Reviews'][user] :
	reviewInfo = data['Review Information'][review];
        stars = float(reviewInfo['stars']);
	bid = reviewInfo['business_id'];

	# Initailize lists and tuples where necesary
	if bid not in businessRevs:
   	    businessRevs[bid] = {};
	    LbidMap[bid] = {};
	    TbidMap[bid] = {};
	    if 'total' not in businessRevs[bid] :
   	        businessRevs[bid]['total'] = 0;
	    if 'positive' not in businessRevs[bid] :
	        businessRevs[bid]['positive'] = 0;

	if stars in pos_list :
	    businessRevs[bid]['positive'] += 1;
	if stars not in LbidMap[bid] :
	    LbidMap[bid][stars] = {};
	if user not in TbidMap[bid] :
	    TbidMap[bid][user] = {};

	# Track reviewID with each business
	# Track information on ratings to each business
        businessRevs[bid]['total'] += 1;
        probability = businessRevs[bid]['positive'] / float(businessRevs[bid]['total']);
        businessRevs[bid]['prob'] = probability; 
	LbidMap[bid][stars][user] = review;
	TbidMap[bid][user][review] = stars;
    #end
#end
print '    %.2f seconds elapsed'%(time.clock()-start);

# Determine and map all users who rated business L as rL
#  and gave business T a positive rating
print '\nCross matching ratings across businesses';
print '    You\'re mapping %d businesses...'%(len(LbidMap));
cross_TrL = []; 
for Tbid in TbidMap :
    for Lbid in LbidMap :
	# Ensure T and L aren't the same business
	if Tbid != Lbid :
    	    for stars in LbidMap[Lbid] :
	        starSum = len(LbidMap[Lbid][stars]);
	        posTbid = 0;
	        for user in LbidMap[Lbid][stars] :
	            if user in TbidMap[Tbid] :
		        rid = LbidMap[Lbid][stars][user];
		        print 'Tbid:%s  Lbid:%s  user:%s  rid:%s'%(Tbid, Lbid, user, rid);
		        reviewRate = TbidMap[Tbid][user][rid];
		        if reviewRate in pos_list :
			    posTbid += 1;
		        # end if reviewRate
	        #end for user
	        numerator = posTbid + 1;
	        denominator = starSum + 1;
	        probability = float(numerator) / denominator;
	        evaluation = {'Tbid':Tbid, 'Lbid':Lbid, 'star':stars, 'prob':probability}
	        cross_TrL.append(evaluation);
	       # sys.stdout.write(str('%.2f'%(time.clock()-start)));
	       # print evaluation;
	    # end for stars
    # end for Lbid
# end for Tbid
print '    %.2f seconds elapsed'%(time.clock()-start);
       
cross_probability = {}; 
print '\nDetermining Advanced Probability...';
print '    This will take a while... you\'re trying to map %d by %d across 5 stars...'%(len(star_info), len(LbidMap));
for Tbid in posTbid :
    cross_probability[Tbid] = {};
    for Lbid in posTbid[Tbid] :
	cross_probability[Tbid][Lbid] = {};
        for starRating in posTbid[Tbid][Lbid] :
	    numerator = 1 + len(posTbid[Tbid][Lbid][starRating]);
	    denominator = 1 + len(star_info[starRating][Lbid]); 
	    probability = float(numerator) / denominator;
    	    cross_probability[Tbid][Lbid][starRating] = probability;
	    print "Tbid: %s   Lbid: %s   Star: %.2f   Prob: %.2f"%(Tbid, Lbid, starRating, probability);
print '    %.2f seconds elapsed'%(time.clock()-start);


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
        if bid not in businessRevs :
            score_lst.append(0.0);
        else :
            score_lst.append(businessRevs[bid]['prob']);
    #end
    fid.write('\n'.join(['%s %.6f %d'%(x[0],x[1],x[2]) for x in zip(bid_lst, score_lst, label_lst)])+'\n');
    fid.close();
#end
print '   %.2f seconds elapsed'%(time.clock()-start);
