import scipy.sparse as sp
import numpy as np
import math
import tfidf
import json
import time
import sys, os
sys.path.append('/home/hltcoe/gsell/tools/python_mods/');
review_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_review.json';
DEFAULT_IDF_UNITTEST = 1.0
test_cond = '12_45'
start = time.clock();

execfile('../load_edge_attr_data.py');
#to_save['Train Reviewer List'] = train_reviewer_lst;
#to_save['Test Reviewer List'] = test_reviewer_lst;
#to_save['Business Information'] = bid_info;
#to_save['Review Information'] = review_info;
#to_save['Reviewer Reviews'] = reviewer_reviews;
#to_save['Reviewed Business List'] = reviewed_bid_lst;
B = len(data['Reviewed Business List'])
R = len(data['Train Reviewer List'])
print '   %.2f seconds elapsed'%(time.clock()-start);

print '\nBuilding index lookups...'
reviewer_idx = {}
business_idx = {}
for n in range(R) : reviewer_idx[data['Train Reviewer List'][n]] = n
for n in range(B) : business_idx[data['Reviewed Business List'][n]] = n
print '   %.2f seconds elapsed'%(time.clock()-start);

print '\nBuilding business review idx'
business_revs = {}
reviewer = []
business = []
review = []
for uid in reviewer_idx :
    for rid in data['Reviewer Reviews'][uid] :
        reviewInfo = data['Review Information'][rid];
        bid = reviewInfo['business_id'];

        # maps the text only for each business within dictionary business_revs
        if bid not in business_revs :
            business_revs[bid] = {};
        if rid not in business_revs[bid] :
            business_revs[bid][rid] = reviewInfo['text'];

        # Used for invoking new csr_matrix 
        reviewer.append(reviewer_idx[uid])
        business.append(business_idx[bid])
        review.append(reviewInfo['text'])
A = sp.csr_matrix((review, (reviewer, business)),shape=[B,R])
A.sort_indices()
B = A.dot(A.T)
print '    %.2f seconds elapsed'%(time.clock()-start);


# Try parsing the review 'text' into the tfidf
total_tfidf = tfidf.TfIdf("tfidf_teststopwords.txt")
for bid in business_revs:
    # Create a new TF-IDF instance for each business reviewed
    business_tfidf = tfidf.TfIdf("tfidf_teststopwords.txt")
    business_doc = ''
    for rid in business_revs[bid] :
        reviewinfo = business_revs[bid][rid]
        business_tfidf.add_input_str(reviewinfo)
        business_doc += reviewinfo 	# Each business is a document for total_tfidf
    total_tfidf.add_input_str(business_doc)

# Prints the TF-IDF scores of 10 top for full dataset
tokens = total_tfidf.return_tokens()
keywords = total_tfidf.get_doc_keywords()
for word in keywords[:10] : 
    print "\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens[word[0]],business_tfidf.get_idf(word[0]),word[1])
print '   %.2f minutes elapsed -- (%.2f sec)'%((time.clock()-start)/60, time.clock()-start);

