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
print 'Clock restart at %.2f seconds elapsed'%(time.clock()-start);

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
print '\nCompiling review ratings';
business_revs = {}; 
star_info = {};
r_pos = [];
r_neg = [];
for uid in reviewer_idx :
    for rid in data['Reviewer Reviews'][uid] :
	reviewInfo = data['Review Information'][rid];
        stars = float(reviewInfo['stars']);
	bid = reviewInfo['business_id'];
        reviewID = reviewInfo['review_id'];

	# Initailize lists and tuples where necesary
	if bid not in business_revs :
   	    business_revs[bid] = {};
	    star_info[bid] = {};
	    star_info[bid]['1star'] = [];
	    star_info[bid]['2star'] = [];
	    star_info[bid]['3star'] = [];
	    star_info[bid]['4star'] = [];
	    star_info[bid]['5star'] = [];
	if reviewID not in business_revs[bid] : 
	    business_revs[bid][reviewID] = [];
        business_revs[bid][reviewID].append(rid);

	# Maintain list of positive and negative reviews
        if stars in pos_list :
            r_pos.append(rid);
        elif stars in neg_list :
            r_neg.append(rid);

	# Maintain a count of each rating for each business
	if stars == 1.0 :
	    star_info[bid]['1star'].append(uid);
	elif stars == 2.0 :
	    star_info[bid]['2star'].append(uid);
	elif stars == 3.0 :
	    star_info[bid]['3star'].append(uid);
	elif stars == 4.0 :
	    star_info[bid]['4star'].append(uid);
	elif stars == 5.0 :
	    star_info[bid]['5star'].append(uid);
	else :
	    print "This isn't doing what you think its doing.... ";
    #end
#end
print '    %.2f seconds elapsed'%(time.clock()-start);


print '\nCross matching ratings across businesses';
rating_map = {};
probRtRl = {};
for bid in star_info :
    print bid;
    for starRating in star_info[bid] :
	for uid in star_info[bid][starRating] :
            for rid in data['Reviewer Reviews'][uid] :
                reviewInfo = data['Review Information'][rid];
                secondStar = float(reviewInfo['stars']);
                secondBid = reviewInfo['business_id'];
		if secondBid != bid :

		    if secondBid not in rating_map :
		        rating_map[secondBid] = {};
		    if secondStar in pos_list :
		        # We want to keep track of this
			if starRating not in rating_map[secondBid] :
			    rating_map[secondBid][starRating] = [];
			rating_map[secondBid][starRating].append(bid);
#	        print "        secondBid: %s  star: %s  bid: %s"%(secondBid, starRating, bid);
		# end if
	    # end for rid
	# end for uid
    # end for star
# end for bid
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
