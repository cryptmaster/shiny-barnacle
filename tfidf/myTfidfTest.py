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


def printTime() :
    timeElapse = time.clock()-start 
    print '\t%.2f seconds elapsed -- %.2f minutes'%(timeElapse, timeElapse/60)


def initialize() :
    print '\nBuilding index lookups...'
    global reviewer_idx
    global business_idx
    for n in range(R) : reviewer_idx[data['Train Reviewer List'][n]] = n
    for n in range(B) : business_idx[data['Reviewed Business List'][n]] = n
    printTime()


# Build sparse matrix 'A' with 'business X reviewer = review' info
def buildIndex() :
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
            reviewText = reviewInfo['text']
            if len(reviewText) > 10 :
                review_idx[reviewCounter] = reviewText
            # Values for the CSR Matrix
            reviewer.append(reviewer_idx[uid]) # r
            business.append(business_idx[bid]) # c
            review.append(reviewCounter)       # d
    A = sp.csr_matrix((review, (business, reviewer)),shape=[B,R])
    A.sort_indices()
    printTime()
    return A


# For each review written to business in test_lst
#   Push each review's text into a compiled "document"
#   then add "doc" as input_str to appropriate TF-IDF instance
def reviewTFIDF(rTFIDF, lst) :
    rDoc = ''
    for review in lst:
        if review in review_idx :
            rDoc += review_idx[review]
    rTFIDF.add_input_str(rDoc)
    return rTFIDF


# Obtain keywords for reviewer TFIDF and publish to .score files
def writeScores() :
    print 'Running evaluation...';
    score_dir = 'scores/TFIDF_%s'%(test_cond);
    os.system('mkdir -p %s'%(score_dir));
    os.system('rm %s/*'%(score_dir));
    here = os.path.dirname(os.path.realpath(__file__));
    for reviewer in test_reviewer_lst :
        [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,reviewer),business_idx);
        outfile = '%s/%s.scores'%(score_dir,reviewer);
        fid = open(outfile,'w');
        print "\nOPENING file %s"%(outfile)
    
        reviewer_tfidf = tfidf.TfIdf("tfidf_teststopwords.txt")	# TFIDF instance for each reviewer
        for (b,i,l) in test_lst :
            # Get the list of reviews given to business in test_lst
            train_reviews = A.getrow(business_idx[b]).tocoo().data
            reviewer_tfidf = reviewTFIDF(reviewer_tfidf, train_reviews)
    
        # Summary of information for the reviewer from all TEST review
        tokens = reviewer_tfidf.return_tokens()
        keywords = reviewer_tfidf.get_doc_keywords()
        fid.write("\n\nReviewer" + str(reviewer) + "\twith test reviews of " + str(len(test_lst)) + "\n")
        for word in keywords[:10] : 
            fid.write("\n\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens[word[0]],reviewer_tfidf.get_idf(word[0]),word[1]))
        fid.write("\n")
        fid.close()
    printTime()


reviewer_idx = {}
business_idx = {}
review_idx = {} 
execfile('../load_edge_attr_data.py');
B = len(data['Reviewed Business List'])
R = len(data['Train Reviewer List'])
D = len(data['Reviewer Reviews'])

initialize()
A = buildIndex()
writeScores()


# EOF
