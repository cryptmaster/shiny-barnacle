import json
import scipy.sparse as sp
import numpy as np
import pickle
import time
import sys

start = time.clock();
business_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_business.json';
review_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_review.json';

print 'Loading reviewer lists...'
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Loading businesses...'
business_str = [x for x in open(business_file).read().split('\n') if len(x)>0];
bid_info = {};
for l in business_str :
    obj = json.loads(l);
    bid_info[obj['business_id']] = obj;
#end
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Loading reviewers and reviews...'
test_reviewer_lst = [x for x in open('lists_%s/test_reviewers.lst'%(test_cond)).read().split('\n') if len(x)>0]
review_str = [x for x in open(review_file).read().split('\n') if len(x)>0];
reviewer_lst = set([]);
review_info = {};
total_review_cnt = 0;
for l in review_str :
    obj = json.loads(l);
    reviewer_lst.add(obj['user_id']);
    review_info[obj['review_id']] = obj;
#end
train_reviewer_lst = sorted(list(reviewer_lst-set(test_reviewer_lst)));
reviewer_lst = sorted(list(reviewer_lst));
review_lst = sorted(review_info.keys());

reviewer_reviews = {};
for r in train_reviewer_lst+test_reviewer_lst : reviewer_reviews[r] = []; #end
for r in review_lst :
    reviewer_reviews[review_info[r]['user_id']].append(r);
#end
print '   %.2f seconds elapsed'%(time.clock()-start);


print 'Linking reviewers to reviewed businesses...'
bid_reviewers = {};
bid_reviews = {};
bid_lst = sorted(bid_info.keys());
for b in bid_lst :
    bid_reviewers[b] = [];
    bid_reviews[b] = [];
#end
for r in train_reviewer_lst :
    for rr in reviewer_reviews[r] :
        bid_reviewers[review_info[rr]['business_id']].append(r);
        bid_reviews[obj['business_id']].append(rr);
    #end
#end
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Creating reviewer index lookup...'
train_reviewer_ndx = {};
cnt = 0;
for r in train_reviewer_lst :
    train_reviewer_ndx[r] = cnt;
    cnt += 1;
#end
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Removing businesses with no reviews...'
reviewed_bid_lst = [];
for b in bid_lst :
    if len(bid_reviewers[b]) > 0 : reviewed_bid_lst.append(b); #end
#end
reviewed_bid_lst.sort();
print '   Reduced from %d to %d'%(len(bid_lst),len(reviewed_bid_lst));
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Loading into data dictionary...'
data = {};
data['Train Reviewer List'] = train_reviewer_lst;
data['Test Reviewer List'] = test_reviewer_lst;
data['Business Information'] = bid_info;
data['Review Information'] = review_info;
data['Reviewer Reviews'] = reviewer_reviews;
data['Reviewed Business List'] = reviewed_bid_lst;
#pickle.dump(to_save,open('projects/pickle/edge_attr_task_data.pkl','w'));
print '   %.2f seconds elapsed'%(time.clock()-start);

