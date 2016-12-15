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
B = len(data['Reviewed Business List'])
R = len(data['Train Reviewer List'])
D = len(data['Reviewer Reviews'])
print '   %.2f seconds elapsed'%(time.clock()-start);

print '\nBuilding index lookups...'
reviewer_idx = {}
business_idx = {}
review_idx = {} 
for n in range(R) : reviewer_idx[data['Train Reviewer List'][n]] = n
for n in range(B) : business_idx[data['Reviewed Business List'][n]] = n
#for n in range(D) : review_idx[n] = data['Review Information'][data['Reviewer Reviews'][n]]['text']
print '   %.2f seconds elapsed'%(time.clock()-start);

print '\nBuilding business review idx'
reviewer = []
business = []
review = []
for uid in reviewer_idx :
    for rid in data['Reviewer Reviews'][uid] :
        reviewInfo = data['Review Information'][rid]
        bid = reviewInfo['business_id']

        # make unique ID for accessing text reviews between bid x rid
        reviewCounter = int(''.join([str(business_idx[bid]),'d',str(reviewer_idx[uid])]))
        reviewtext = reviewInfo['text']
        if len(reviewtext) > 10 :
            review_idx[reviewCounter] = reviewInfo['text']
        # Values for the CSR Matrix
        reviewer.append(reviewer_idx[uid]) # r
        business.append(business_idx[bid]) # c
        review.append(reviewCounter)      # d
        #print 'c: ' + str(business_idx[bid]) + '\tr: ' + str(reviewer_idx[uid]) + '\td: ' + reviewIdx
A = sp.csr_matrix((review, (business, reviewer)),shape=[B,R])
#A.sort_indices()
#At = A.dot(A.T)
print '    %.2f seconds elapsed'%(time.clock()-start);

print '\n Determining TF-IDF across all businesses'
# Keep a 'total' TF-IDF for all businesses
total_tfidf = tfidf.TfIdf("tfidf_teststopwords.txt")
for business in range(B) :
    # Create a TF-IDF instance for each business reviewed
    business_tfidf = tfidf.TfIdf("tfidf_teststopwords.txt")
    business_doc = ''
    businessReviews = A.getrow(business).tocoo().data
    # Iterate through all reviews given to business
    for review in businessReviews :
        # Get review text and save it for TF-IDF
        reviewinfo = review_idx[review]
        print str(business) + ' ' + str(review) + ' ' + reviewinfo
        business_tfidf.add_input_str(reviewinfo)
        business_doc += reviewinfo 
    # Add all reviews for business as single document
    total_tfidf.add_input_str(business_doc)
print '    %.2f seconds elapsed'%(time.clock()-start);

# Prints the TF-IDF scores of 10 top for full dataset
tokens = total_tfidf.return_tokens()
keywords = total_tfidf.get_doc_keywords()
for word in keywords[:10] : 
    print "\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens[word[0]],business_tfidf.get_idf(word[0]),word[1])
print '   %.2f minutes elapsed -- (%.2f sec)'%((time.clock()-start)/60, time.clock()-start);

