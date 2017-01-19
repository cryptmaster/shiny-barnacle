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


# For each test reviewer, use the 'train_lst' to score 'test_lst'
def evaluate_data() :
    for reviewer in test_reviewer_lst :
        [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,reviewer),business_idx);
        print '\n\nProcessing reviewer %s'%(str(reviewer))

        X_train = []
        y_train = []
        for (b,i,l) in train_lst :
            reviews = build_review_str(A[l].getrow(business_idx[b]).tocoo().data)
            X_train.append(reviews)
            y_train.append(l)

        X_test = []
        y_test = []
        for (b,i,l) in test_lst :
            reviews = []
            for s in [-1,1] :
                reviews.append(build_review_str(A[s].getrow(business_idx[b]).tocoo().data))
            reviews = ' '.join(reviews)
            X_test.append(reviews)
            y_test.append(l)
            scoredTFIDF = build_TFIDF(reviews, False)

        bagOfWords(X_train, y_train, X_test, y_test)
        printTime()


def trainData(X_train, y_train) :
    vect = TfidfVectorizer(ngram_range=(1,3), stop_words='english')
    X_train = vect.fit_transform(X_train)
    feature_names = vect.get_feature_names()

    # Logistic Regression scores determined for training data
    scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
    print "Mean Train Score: " + str(np.mean(scores))

    # Exhaustive search over specified parameter values for an estimator
    # Fits training data to 'grid' & returns score of best_estimator on the
    # ..left out data and the parameter setting that gave the best results
    # ..on the hold out data
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print ("Best cross-validation score: ", grid.best_score_)
    print ("Best parameters: ", grid.best_params_)
    
    return vect, grid


# Bag-of-word example from 'Introduction to Machine Learning with Python' by O'Reily
def bagOfWords(X_train, y_train, X_test, y_test) :
    vect, grid = trainData(X_train, y_train)

    # Transform test documents to document-term matrix
    # Returns the mean accuracy on the given test data and labels
    X_test = vect.transform(X_test)
    print "Test score: " + str(grid.score(X_test, y_test))


# TF-IDF is determined for words in the document 'corpus' 
# Dictionary is returned with word:value sorted in desending order by TF-IDF
# 'topicModel' is a boolean to determine if the topic model should be determined
# ..and returned during this instance
def build_TFIDF(corpus, topicModel) :
    vect = TfidfVectorizer(
                ngram_range=(1,3),
                stop_words='english'
            )
    tfidf = vect.fit_transform(corpus)
    feature_names = vect.get_feature_names()
    idf = vect.idf_

    vectorDict = dict(zip(feature_names, idf))

    if topicModel :
        display_topics(tfidf, feature_names)
    return sorted(vectorDict.items(), key=operator.itemgetter(1), reverse=True)


# Create a dictionary of words from review
def build_dictionary(review) :
    dictionary = []
    if len(review) > 0 :
        sortedDict = build_TFIDF(review, False)
        for word in sortedDict :
            dictionary.append(word[0])
    return dictionary


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


# Determines Topic Model using NMF from TF-IDF ('model') 'feature_names'
# Prints 'no_top_words' for each of the 'no_topics' topics 
def display_topics(model, feature_names) :
    no_topics = 5
    no_top_words = 10
    nmf = NMF(n_components=no_topics, random_state=1).fit(model)
    for topic_idx, topic in enumerate(nmf.components_):
        print "Topic #%d:" %(topic_idx)
        print " ,".join([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]])

# Prints scored TF-IDF vectors for test businesses of each user
def printToFile() :
    score_dir = 'TFIDFscores'
    os.system('rm %s/*'%(score_dir));
    os.system('mkdir -p %s'%(score_dir));
    here = os.path.dirname(os.path.realpath(__file__));

    for reviewer in test_reviewer_lst :
        [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,reviewer),business_idx);
        bid_lst = [];
        score_lst = [];
        label_lst = [];
        print '\n\nProcessing reviewer %s'%(str(reviewer))

        for (b,i,l) in test_lst :
            outfile = '%s/%s/%s.scores'%(score_dir,reviewer,b)
            fid = open(outfile,'w');
            reviews = []
            for s in [-1,1] :
                reviews.append(build_review_str(A[s].getrow(business_idx[b]).tocoo().data))
            reviews = ' '.join(reviews)
            scoredTFIDF = build_TFIDF(reviews, False)
            print "writing below to file " + str(outfile)
            print scoredTFIDF
            fid.write(scoredTFIDF)
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
