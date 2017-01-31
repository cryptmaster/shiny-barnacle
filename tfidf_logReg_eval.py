# File: tfidf_sample_01.py
# 
# This script goes over the Yelp data stored at variable review_file to generate
# ..TF-IDF values for each document, where a document is defined as all reviews
# ..given to a business. TF-IDF is determined using feature_extraction from the 
# ..scikit-learn.org sklearn package (v0.16)

import json
import scipy.sparse as sp
import numpy as np
import pickle
import time
import util_functions as util
import sys, os
import math
import operator

from sklearn import linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline


sys.path.append('/home/hltcoe/gsell/tools/python_mods/');
review_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_review.json';
DEFAULT_IDF_UNITTEST = 1.0
test_cond = '12_45'
start = time.clock();
pos_lst = [4,5];
neg_lst = [1,2];


pos_lst = []
neg_lst = []
reviewer_idx = {}
business_idx = {}
review_idx = {} 
execfile('load_edge_attr_data.py');
print '\nBuilding index lookups...'
B = len(data['Reviewed Business List'])
R = len(data['Train Reviewer List'])
D = len(data['Reviewer Reviews'])
for n in range(R) : reviewer_idx[data['Train Reviewer List'][n]] = n
for n in range(B) : business_idx[data['Reviewed Business List'][n]] = n


# Print time elapse in seconds && minutes
def printTime() :
    timeElapse = time.clock()-start 
    print '\t%.2f seconds elapsed -- %.2f minutes'%(timeElapse, timeElapse/60)
printTime()


print '\nBuilding business review idx'
r = [] 	#reviewer 
c = []	#business 
d = []	#review 
for uid in reviewer_idx :
    for rid in data['Reviewer Reviews'][uid] :
        reviewInfo = data['Review Information'][rid]
        bid = reviewInfo['business_id']
        reviewText = reviewInfo['text']
        stars = float(reviewInfo['stars']) 
    	reviewCounter = int(''.join([str(business_idx[bid]),str(reviewer_idx[uid])]))

        review_idx[reviewCounter] = reviewText
    	d.append(reviewCounter)		# d
        r.append(reviewer_idx[uid]) 	# r
        c.append(business_idx[bid]) 	# c
A = sp.csr_matrix((d,(c,r)),shape=[B,R])
printTime()


###################################################################################

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

###################################################################################

datetime = time.strftime("%Y%m%d_%H%M")
score_dir = 'projects/scores/TFIDF_%s_%s'%(test_cond,datetime)
os.system('mkdir -p %s'%(score_dir));
os.system('rm %s/*'%(score_dir));
here = os.path.dirname(os.path.realpath(__file__));

for reviewer in test_reviewer_lst :
    [train_lst,test_lst] = util.read_key('lists_%s/%s.key'
                                          %(test_cond,reviewer),business_idx)
    outfile = '%s/%s.status'%(score_dir,reviewer);
    fid = open(outfile,'w')

    print '\n\nProcessing reviewer %s'%(str(reviewer))
    text_train = []
    y_train = []
    for (b,i,l) in train_lst :
        text_train.append(build_review_str(A.getrow(business_idx[b]).tocoo().data))
        y_train.append(l)
#    scores = cross_val_score(LogisticRegression(), text_train, y_train, cv=5)
#    print "Mean cross-validation accuracy: {:.2f}".format(np.mean(scores))
#    grid = trainData(X_train, y_train)
    pipe = make_pipeline(TfidfVectorizer(stop_words="english"), LogisticRegression())
    param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10],
                  'tfidfvectorizer__ngram_range': [(1,1), (1,2)]} 
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(text_train, y_train)
    bestScore = "Best cross-validation score: \t{:.2f}".format(grid.best_score_)
    bestParams = "Best parameters: \n\t{}".format(grid.best_params_)
    fid.write('\n'+bestScore+'\n'+bestParams)
    print bestScore
    print bestParams
    fid.write("\nGrid scores on development set:")
    for params, mean_score, scores in grid.grid_scores_:
        fid.write("\n\t%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() *2, params))


    # Running test set, scoring, print to files
    text_test = []
    y_test = []
    bid_lst = []
    for (b,i,l) in test_lst :
        bid_lst.append(b)
        text_test.append(build_review_str(A.getrow(business_idx[b]).tocoo().data))
        y_test.append(l)

    testScore = '\nGeneralized performance assessment on test: {:.2f}'.format(grid.score(text_test, y_test))
    print testScore
    fid.write(testScore)
    fid.write('\n')
    y_true, y_pred = y_test, grid.predict_proba(text_test)[:,1]
#    testReport = classification_report(y_true, y_pred)
#    print testReport
#    fid.write(testReport)
    fid.close()


    outfile = '%s/%s.scores'%(score_dir,reviewer);
    fid = open(outfile,'w')
    fid.write('\n'.join(['%s %.6f %d'%(x[0],x[1],x[2]) for x in zip(bid_lst, y_pred, y_true)])+'\n')
    fid.close()
    printTime() 
printTime() 

print'python score_rank_list.py -l lists_%s/ -s %s'%(test_cond,score_dir);
os.system('python score_rank_list.py -l lists_%s/ -s %s'%(test_cond,score_dir));
