import json
import scipy.sparse as sp
import numpy as np
import itertools
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


############################
# Determine and map all users who rated business L as rL
#  and gave business T a positive rating
############################
def crossMatch(TbidMap) :
    global fullSummation; 
    print '    You\'re mapping %d by %d businesses...'%(len(TbidMap), len(LbidMap));
    Tcount = 0;
    totalbid = len(LbidMap);
    for Tbid in TbidMap :
        summation[Tbid] = 0; 
        Tcount += 1;
        Lcount = 0;
        for Lbid in LbidMap :
    	    Lcount += 1;
	    if Tbid != Lbid :
	        # Get numer of reviews at EACH STAR rate for L
    	        for stars in LbidMap[Lbid] :
	            starSum = len(LbidMap[Lbid][stars]);
	            posTbid = 0;
		    # For each review check if user rated the Tbid
		    for Lreview in LbidMap[Lbid][stars] :
		        user = reviewMap[Lreview]['user'];
		        if user in TbidMap[Tbid] :
			    # user rev'd Tbid, get their Trid & see if gave Tbid pos rev
			    for Trid in TbidMap[Tbid][user] :
			        Tstar = reviewMap[Trid]['stars'];
			        if Tstar in pos_list :
				    posTbid += 1;
		    # End for Lreview
	            crossProb = float(posTbid + 1) / (starSum + 1);
		    simpleProb = businessRevs[Tbid]['prob'];
		    summationItem = detProbability(simpleProb, crossProb);

	   	    summation[Tbid] += summationItem; 
		    fullSummation += summationItem;
	            sys.stdout.write(str('%.2f'%(time.clock()-start)));
	            print ' | bid:%d:%d/%d | prob a:%.3f s:%.3f | summ T:%.1f F:%.1f'%(Tcount, Lcount, totalbid, printcross, printsimple, summation[Tbid], fullSummation);
# end crossMatch

############################
# Probability calculated in crossMatch
############################
def detProbability(simpleProb, crossProb) :
    if simpleProbability > 0.0001 :
        simpleProbability = float(math.log(simpleProbability));
    else :
        simpleProbabiliy = 0.0;

    if crossProbability > 0.0001 :
        crossProbability = float(math.log(crossProbability)); 
    else :
        simpleProbability = 0.0;
    summationItem = crossProbability - float(simpleProbability);
    return summationItem;

############################
############################
# Start of program
############################
############################
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

############################
# Create list of positive and negative reviews
############################
print '\nCompiling Review Maps... ';
businessRevs = {};
LbidMap = {};
TbidMap = {};
reviewMap = {};
for user in reviewer_idx :
    for review in data['Reviewer Reviews'][user] :
	reviewInfo = data['Review Information'][review];
        stars = float(reviewInfo['stars']);
	bid = reviewInfo['business_id'];

	# Initailize lists and tuples where necesary
	reviewMap[review] = {};
	if bid not in businessRevs:
   	    businessRevs[bid] = {};
	    LbidMap[bid] = {};
	    TbidMap[bid] = {};
	    if 'total' not in businessRevs[bid] :
   	        businessRevs[bid]['total'] = 0;
	    if 'positive' not in businessRevs[bid] :
	        businessRevs[bid]['positive'] = 0;
	if stars not in LbidMap[bid] :
	    LbidMap[bid][stars] = [];
	if user not in TbidMap[bid] :
	    TbidMap[bid][user] = [];

	if stars in pos_list :
	    businessRevs[bid]['positive'] += 1;
	reviewMap[review]['stars'] = stars;
	reviewMap[review]['user'] = user;
	reviewMap[review]['bid'] = bid;
        businessRevs[bid]['total'] += 1;
        probability = businessRevs[bid]['positive'] / float(businessRevs[bid]['total']);
        businessRevs[bid]['prob'] = probability; 
	LbidMap[bid][stars].append(review);
	TbidMap[bid][user].append(review);
    #end
#end
print '    %.2f seconds elapsed'%(time.clock()-start);

#############################
# make-shift divide & conquor
#############################
begin = 0;
end = 0;
summation = {};
fullSummation = 0;
print '\nCross matching ratings across businesses';
while end <= len(TbidMap) :
    end += 100;
    if end > len(TbidMap) :
        end = len(TbidMap);

    Tsubset = dict(itertools.islice(TbidMap.iteritems(), begin, end));
    crossMatch(Tsubset);
    begin = end + 1;
print '    %.2f seconds elapsed'%(time.clock()-start);


#############################
## Enter data to .scores files
#############################
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
	probability = businessRevs[bid]['prob'];
	if probability > 0.0001 :
		probability = float(math.log(businessRevs[bid]['prob']));
	score1 = probability + float(fullSummation);
	score1_lst.append(score1);
	score2 = probability + float(summation[bid]);
	score2_lst.append(score2);
	probability = probability * 100;
	print 'Bid:%s  | S1:%.6f  S2:%.6f'%(bid, score1, score2);
    #end
    fid.write('\n'.join(['%s %.6f %.6f %d'%(x[0],x[1],x[3],x[4]) for x in zip(bid_lst, score1_lst, score2_lst, label_lst)])+'\n');
    fid.close();
#end
print '   %.2f seconds elapsed'%(time.clock()-start);




