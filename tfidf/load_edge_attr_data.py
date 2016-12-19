# Modified version of the original to provid just necessary data for 
# TF-IDF with goal of reducing overhead time
import json
import time
start = time.clock();
business_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_business.json';
review_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_review.json';

print 'Loading businesses...'
business_str = [x for x in open(business_file).read().split('\n') if len(x)>0];
bid_info = {};
for l in business_str :
    obj = json.loads(l);
    bid_info[obj['business_id']] = obj;
print '   %.2f seconds elapsed'%(time.clock()-start);


print 'Loading reviewers and reviews...'
test_reviewer_lst = [x for x in open('lists_%s/test_reviewers.lst'%(test_cond)).read().split('\n') if len(x)>0]
review_str = [x for x in open(review_file).read().split('\n') if len(x)>0];

reviewer_lst = set([]);
review_info = {};
for l in review_str :
    obj = json.loads(l);
    reviewer_lst.add(obj['user_id']);
    review_info[obj['review_id']] = obj;
train_reviewer_lst = sorted(list(reviewer_lst-set(test_reviewer_lst)));
review_lst = sorted(review_info.keys());

reviewer_reviews = {};
for r in train_reviewer_lst+test_reviewer_lst : reviewer_reviews[r] = []; #end
for r in review_lst : reviewer_reviews[review_info[r]['user_id']].append(r);
print '   %.2f seconds elapsed'%(time.clock()-start);


print 'Linking reviewers to reviewed businesses...'
bid_reviewers = {};
bid_lst = sorted(bid_info.keys());
for b in bid_lst : bid_reviewers[b] = [];
for r in train_reviewer_lst :
    for rr in reviewer_reviews[r] :
        bid_reviewers[review_info[rr]['business_id']].append(r);
print '   %.2f seconds elapsed'%(time.clock()-start);


print 'Removing businesses with no reviews...'
reviewed_bid_lst = [];
for b in bid_lst :
    if len(bid_reviewers[b]) > 0 : reviewed_bid_lst.append(b); #end
reviewed_bid_lst.sort();
print '   Reduced from %d to %d'%(len(bid_lst),len(reviewed_bid_lst));
print '   %.2f seconds elapsed'%(time.clock()-start);


print 'Loading into data dictionary...'
data = {};
data['Train Reviewer List'] = train_reviewer_lst;
data['Reviewed Business List'] = reviewed_bid_lst;

data['Test Reviewer List'] = test_reviewer_lst;
data['Business Information'] = bid_info;
data['Review Information'] = review_info;
data['Reviewer Reviews'] = reviewer_reviews;
print '   %.2f seconds elapsed'%(time.clock()-start);

