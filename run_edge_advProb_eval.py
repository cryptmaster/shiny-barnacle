import json
import scipy.sparse as sp
import numpy as np
import math
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
	LbidMap[bid][stars][review] = user;
	TbidMap[bid][user][review] = stars;
    #end
#end
print '    %.2f seconds elapsed'%(time.clock()-start);

# Determine and map all users who rated business L as rL
#  and gave business T a positive rating
print '\nCross matching ratings across businesses';
print '    You\'re mapping %d businesses...'%(len(LbidMap));
cross_TrL = {}; 
counter = 0;
Lcount = 0;
summation = {};
fullSummation = 0;
totalbid = len(LbidMap);
for Tbid in TbidMap :
    cross_TrL[Tbid] = {};
    summation[Tbid] = 0; 
    counter += 1;
    for Lbid in LbidMap :
	Lcount += 1;
	# Ensure T and L aren't the same business
	if Tbid != Lbid :
	    cross_TrL[Tbid][Lbid] = {};
	    # Get numer of reviews at EACH STAR rate for L
    	    for stars in LbidMap[Lbid] :
	        starSum = len(LbidMap[Lbid][stars]);
	        posTbid = 0;
		# For each review check if user rated the Tbid
		for Lreview in LbidMap[Lbid][stars] :
		    user = LbidMap[Lbid][stars][Lreview];
		    if user in TbidMap[Tbid] :
			# user rev'd Tbid, get their Trid 
			#  and see if they gave Tbid a pos rev
			for Trid in TbidMap[Tbid][user] :
			    # Currently this does not account for multiple reviews
			    #  given by the same person. Just want to get this 
			    #  working and then I'll minimize this
			    Tstar = TbidMap[Tbid][user][Trid];
			    if Tstar in pos_list :
				posTbid += 1;
	        numerator = posTbid + 1;
	        denominator = starSum + 1;
	        crossProbability = float(numerator) / denominator;
		simpleProbability = businessRevs[Tbid]['prob'];
	        cross_TrL[Tbid][Lbid][stars] = crossProbability;
		summationItem = math.log(crossProbability) - float(math.log(simpleProbability));
	   	summation[Tbid] += summationItem; 
		fullSummation += summationItem;
	        sys.stdout.write(str('%.2f'%(time.clock()-start)));
	        print ' | Tbid:%d/%d Lbid:%d/%d | advProb:%.3f simProb:%.3f'%(counter, totalbid, Lcount, totalbid, crossProbability, simpleProbability);
print '    %.2f seconds elapsed'%(time.clock()-start);
       

print 'Running evaluation...';
score_dir = 'scores/AdvancedProbability_%s'%(test_cond);
os.system('mkdir -p %s'%(score_dir));
os.system('rm %s/*'%(score_dir));
here = os.path.dirname(os.path.realpath(__file__));
for reviewer in test_reviewer_lst :
    [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,reviewer),business_idx);
    outfile = '%s/%s.scores'%(score_dir,reviewer);
    fid = open(outfile,'w');
    bid_lst = [];
    score1_lst = [];
    score2_lst = [];
    label_lst = [];

    # For each reviewer, find the buses reviewed
    # and write the bid, prob/score, and a label to file
    # such that all buses rev'd by UID are in single file
    for value in test_lst :
        bid = value[0];
        bid_lst.append(bid);
        label_lst.append(value[2]);
	score1 = math.log(businessRevs[bid]['prob']) + float(fullSummation);
	score1_lst.append(score1);
	score2 = math.log(businessRevs[bid]['prob']) + float(summation[bid]);
	score2_lst.append(score2);
	print 'Bid:%s  Score1:%.6f  Score2:%.6f'%(bid, score1, score2);
    #end
    fid.write('\n'.join(['%s %.6f %.6f %d'%(x[0],x[1],x[2],x[3]) for x in zip(bid_lst, score1_lst, score2_lst, label_lst)])+'\n');
    fid.close();
#end
print '   %.2f seconds elapsed'%(time.clock()-start);
print "Somehow we made it to the end. Congrats"
