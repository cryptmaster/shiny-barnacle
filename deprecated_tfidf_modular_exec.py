# This script goes over the Yelp data stored at variable review_file to generate
# ..TF-IDF values for each document, where a document is defined as all reviews
# ..given to a business. TF-IDF is determined using feature_extraction from the 
# ..scikit-learn.org sklearn package (v0.16)
#
# Example run instance: 'python tfidf_onUser_byBusiness.py posneg3out'

import json
import scipy.sparse as sp
import numpy as np
import pickle
import time
import util_functions as util
import sys, os
import math
import operator
import tfidf
import sklearn.feature_extraction.text as text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from sklearn import linear_model


sys.path.append('/home/hltcoe/gsell/tools/python_mods/');
review_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_review.json';
DEFAULT_IDF_UNITTEST = 1.0
test_cond = '12_45'
start = time.clock();
edge_type = sys.argv[1]


# Print time elapse in seconds && minutes
def printTime() :
    timeElapse = time.clock()-start 
    print '\t%.2f seconds elapsed -- %.2f minutes'%(timeElapse, timeElapse/60)


# Build required indicies for reference
def initialize() :
    print '\nBuilding index lookups...'
    global reviewer_idx
    global business_idx
    global pos_lst
    global neg_lst
    for n in range(R) : reviewer_idx[data['Train Reviewer List'][n]] = n
    for n in range(B) : business_idx[data['Reviewed Business List'][n]] = n

    if 'posneg' in edge_type :
        if '3neg' in edge_type :
            pos_lst = [4,5];
            neg_lst = [1,2,3];
        elif '3pos' in edge_type :
            pos_lst = [3,4,5];
            neg_lst = [1,2];
        elif '3out' in edge_type :
            pos_lst = [4,5];
            neg_lst = [1,2];
        elif '234out' in edge_type :
            pos_lst = [5];
            neg_lst = [1];
        else :
            print 'Unknown edge type '+edge_type;
            sys.exit();
    printTime()


# Build sparse matrix 'A' with 'business X reviewer = review' info
def build_index() :
    print '\nBuilding business review idx'
    r = {} 	#reviewer 
    c = {}	#business 
    d = {}	#review 
    A = {}
    for s in [-1,1] :
        r[s] = []
        c[s] = []
        d[s] = []
    for uid in reviewer_idx :
        for rid in data['Reviewer Reviews'][uid] :
            reviewInfo = data['Review Information'][rid]
            bid = reviewInfo['business_id']
            reviewText = reviewInfo['text']
            stars = float(reviewInfo['stars']) 
            # For viable data, limiting to just reviews with more than 10 characters
            if len(reviewText) > 10 :
		reviewCounter = int(''.join([str(business_idx[bid]),str(reviewer_idx[uid])]))
                review_idx[reviewCounter] = reviewText
                if stars in pos_lst :
		    d[1].append(reviewCounter)		# d
                    r[1].append(reviewer_idx[uid]) 	# r
                    c[1].append(business_idx[bid]) 	# c
                elif stars in neg_lst :
		    d[-1].append(reviewCounter)		# d
                    r[-1].append(reviewer_idx[uid]) 	# r
                    c[-1].append(business_idx[bid]) 	# c
    A[1] = sp.csr_matrix((d[1],(c[1],r[1])),shape=[B,R])
    A[-1] = sp.csr_matrix((d[-1],(c[-1],r[-1])),shape=[B,R])
    printTime()
    return A


# TF-IDF is determined for words in the document 'corpus' 
# Dictionary is returned with word:value sorted in desending order by TF-IDF
# 'topicModel' is a boolean to determine if the topic model should be determined
# ..and returned during this instance
def build_TFIDF(corpus) :
    vect = TfidfVectorizer(
                ngram_range=(1,2),
                stop_words='english'
            )
    tfidf = vect.fit_transform(corpus)
    feature_names = vect.get_feature_names()
    idf = vect.idf_

    vectorDict = dict(zip(feature_names, idf))

    return sorted(vectorDict.items(), key=operator.itemgetter(1), reverse=True)


# Store all reviews from a review_lst as a list of the string reviews
def build_review_lst(review_lst) :
    reviews = []
    for review in review_lst:
        if review in review_idx :
            reviews.append(review_idx[review])
    return reviews


# Compact all reviews in a review_lst as a single string
# Calls build_review_lst and joins all items in list
def build_review_str(review_lst) :
    reviews = build_review_lst(review_lst)
    reviewLine = ' '.join(reviews)
    return reviewLine


# Prints scored TF-IDF vectors for test businesses of each user
def printToFile() :
    score_dir = '/export/projects/vlyzinski/MiniScale2017/TFIDFscores'
    os.system('mkdir -p %s'%(score_dir));
    os.system('rm %s/*'%(score_dir));
    here = os.path.dirname(os.path.realpath(__file__));

    for reviewer in test_reviewer_lst :
        [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,reviewer),business_idx);
        print '\n\nProcessing reviewer %s'%(str(reviewer))

        for (b,i,l) in test_lst :
            posRevs = build_review_lst(A[1].getrow(business_idx[b]).tocoo().data) 
            negRevs = build_review_lst(A[-1].getrow(business_idx[b]).tocoo().data) 
            reviews = posRevs + negRevs
            print 'scoring on ' + str(len(reviews)) + ' reviews'

            if len(reviews) > 0 :
                outfile = '%s/%s.scores'%(score_dir,str(b))
                fid = open(outfile,'w')
                scoredTFIDF = build_TFIDF(reviews)
                wc = 0
                for word in scoredTFIDF :
                    fid.write(str(wc) + '\t')
                    fid.write(word[0].encode('ascii','ignore'))
                    fid.write('\t')
                    fid.write(str(word[1]))
                    fid.write('\n')
                    wc += 1
                fid.close();
        printTime() 


# -----------------MAIN--------------------
pos_lst = []
neg_lst = []
reviewer_idx = {}
business_idx = {}
review_idx = {} 
execfile('../load_edge_attr_data.py');
B = len(data['Reviewed Business List'])
R = len(data['Train Reviewer List'])
D = len(data['Reviewer Reviews'])

initialize()
A = build_index() 
printToFile()
# EOF
