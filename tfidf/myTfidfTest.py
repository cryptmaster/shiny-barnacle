import json
import scipy.sparse as sp
import numpy as np
import pickle
import time
import sklearn.metrics as metrics
import util_functions as util
import sys, os
import math
import tfidf
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
        reviewCounter = int(''.join([str(business_idx[bid]),str(reviewer_idx[uid])]))
        if reviewCounter == 4255788 :
            print reviewInfo['text']
        reviewtext = reviewInfo['text']
        if len(reviewtext) > 10 :
            review_idx[reviewCounter] = reviewInfo['text']
        # Values for the CSR Matrix
        reviewer.append(reviewer_idx[uid]) # r
        business.append(business_idx[bid]) # c
        review.append(reviewCounter)      # d
A = sp.csr_matrix((review, (business, reviewer)),shape=[B,R])
A.sort_indices()
print '    %.2f seconds elapsed'%(time.clock()-start);


## Enter data to .scores files
#############################
print 'Running evaluation...';
score_dir = 'scores/TFIDF_%s'%(test_cond);
os.system('mkdir -p %s'%(score_dir));
os.system('rm %s/*'%(score_dir));
here = os.path.dirname(os.path.realpath(__file__));
reviewCount = 0
for reviewer in test_reviewer_lst :
    [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,reviewer),business_idx);
    outfile = '%s/%s.scores'%(score_dir,reviewer);
    fid = open(outfile,'w');
    reviewCount += 1
    print "\nOPENING file %s"%(outfile)

    reviewer_tfidf = tfidf.TfIdf("tfidf_teststopwords.txt")	# TFIDF instance for each reviewer
    for (b,i,l) in test_lst :
        # Get the list of reviews given to business in test_lst
        train_reviews = A.getrow(business_idx[b]).tocoo().data
        print '\tFor business ' + str(b) + '\t' + 'Train businesses for test: ' + str(len(train_reviews))
        fid.write('\nFor business ' + str(b) + '\t' + 'Train businesses for test: ' + str(len(train_reviews)))

        train_tfidf = tfidf.TfIdf("tfidf_teststopwords.txt")	# TFIDF instance for each business
        reviewer_doc = ''
        # For each review written to business in test_lst
        #   Push each review's text into a compiled "document"
        #   then add "doc" as input_str to appropriate TF-IDF instance
        for review in train_reviews :
            train_doc = ''
            if review in review_idx :
                train_doc += review_idx[review]
                reviewer_doc += review_idx[review]
            train_tfidf.add_input_str(train_doc)
        reviewer_tfidf.add_input_str(reviewer_doc)

        # Summary of information for each test businesses' TFIDF
        tokens = train_tfidf.return_tokens()
        keywords = train_tfidf.get_doc_keywords()
        for word in keywords[:5] : 
            fid.write("\n\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens[word[0]],train_tfidf.get_idf(word[0]),word[1]))

    # Summary of information for the reviewer from all TEST review
    tokens = reviewer_tfidf.return_tokens()
    keywords = reviewer_tfidf.get_doc_keywords()
    fid.write("\n\nReviewer" + str(reviewer) + "\twith test reviews of " + str(len(test_lst)) + "\n")
    for word in keywords[:10] : 
        fid.write("\n\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens[word[0]],reviewer_tfidf.get_idf(word[0]),word[1]))
    fid.write("\n")
    fid.close()

    print 'Reviewer %d out of %d --- \t%.2f seconds elapsed'%(reviewCount, len(test_reviewer_lst), time.clock()-start);
print '%.2f seconds elapsed'%(time.clock()-start);
