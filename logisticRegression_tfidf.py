# This script goes over the Yelp data stored at variable review_file to generate
# ..TF-IDF values for each document, where a document is defined as all reviews
# ..given to a business. TF-IDF is determined using feature_extraction from the 
# ..scikit-learn.org sklearn package (v0.16)
#
# Example run instance: 'python tfidf_onUser_byBusiness.py posneg3out'
from __future__ import print_function

import json
import logging
import math
import numpy as np
import operator
from optparse import OptionParser
import pickle
import scipy.sparse as sp
import sys, os
import time
import util_functions as util

from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid 
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn import linear_model, metrics, neighbors

sys.path.append('/home/hltcoe/gsell/tools/python_mods/');
review_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_review.json';
DEFAULT_IDF_UNITTEST = 1.0
test_cond = '12_45'
start = time.clock();

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test.")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class for every classifier.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("This script takes no arguments")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

########################################################################################

pos_lst = []
neg_lst = []
reviewer_idx = {}
business_idx = {}
review_idx = {} 

execfile('load_edge_attr_data.py');
B = len(data['Reviewed Business List'])
R = len(data['Train Reviewer List'])
D = len(data['Reviewer Reviews'])

print('\nBuilding index lookups...')
for n in range(R) : reviewer_idx[data['Train Reviewer List'][n]] = n
for n in range(B) : business_idx[data['Reviewed Business List'][n]] = n
pos_lst = [4,5]
neg_lst = [1,2]

# Print time elapse in seconds && minutes
def printTime() :
    timeElapse = time.clock()-start 
    print('\t%.2f seconds elapsed -- %.2f minutes'%(timeElapse, timeElapse/60))

printTime()


print('\nBuilding business review idx')
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


for reviewer in test_reviewer_lst :
    [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,reviewer),business_idx);
    print('\n\nProcessing reviewer %s' % str(reviewer))

    X_train = []
    y_train = []
    for (b,i,l) in train_lst :
        reviews = build_review_str(A[l].getrow(business_idx[b]).tocoo().data)
        X_train.append(reviews)
        y_train.append(l)
    reviews = np.array(X_train)

    # Exhaustive search over specified parameter values for an estimator
    # Fits training data to 'grid' & returns score of best_estimator on the
    # ..left out data and the parameter setting that gave the best results
    # ..on the hold out data
    print("Extracting features from the training data using a sparse vectorizer")
    if opts.use_hashing :
        vect = HashingVectorizer(stop_words='english', non_negative=True,
                                 n_features=opts.n_features)
        X_train = vectorizer.transform(reviews)
    else :
        vect = TfidfVectorizer(sumblinear_tf=True, max_df=0.5, 
                               stop_words='english')
        X_train = vect.fit_transform(reviews)
    printTime()
    print ("\tn_samples: %d, n_features: %d"% X_train.shape)

    print("Determining Logistic Regression scores for training data")
    scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
    print("\tMean Train Score: " + str(np.mean(scores)))
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print ("\tBest cross-validation score: ", grid.best_score_)
    print ("\tBest parameters: ", grid.best_params_)


    X_test = []
    y_test = []
    for (b,i,l) in test_lst :
        local_reviews = []
        for s in [-1,1] :
            local_reviews.append(build_review_str(A[s].getrow(business_idx[b]).tocoo().data))
        local_reviews = ' '.join(local_reviews)
        X_test.append(local_reviews)
        y_test.append(l)
    reviews = np.array(X_test) 
    scoredTFIDF = build_tfidf(reviews, False)

    # Transform test documents to document-term matrix
    # Returns the mean accuracy on the given test data and labels
    print("Extracting features from the test data using the same vectorizer")
    X_test = vect.transform(reviews)
    printTime()
    print ("\tn_samples: %d, n_features: %d"% X_test.shape)
#    feature_names = np.asarray(vect.get_feature_names())
#    print "Test score: " + str(grid.score(X_test, y_test))
    print ()


    # mapping from integer feature name to original token string
    if opts.use_hashing :
        feature_names = None
    else :
        feature_names = vectorizer.get_feature_names()

    if opts.select_chi2 :
        print("Extracting %d best features by a chi-squared test" %opts.select_chi2)
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        if feature_names:
            # keep selected feature names
            feature_names = [feature_names[i] for i in ch2.get_support(indicies=True)]
        printTime()
        print()

    if feature_names:
        feature_names = np.asarray(feature_names)

#########################################################################################
# Benchmark classifiers
def benchmark(clf) :
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" %(train_time))
    
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time: %0.3fs" %(test_time))

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:  %0.3f" % score)

    if hasattr(clf, 'coef_') :
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None :
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i][-10:])
                toPrint = "%s: %s" % (category, " ".join(feature_names[top10]))
                print(toPrint)
        print()
   
    if otps.print_report :
        print("classification report: ")
        print(metrics.classification_report(y_test, pred, target_names=categories))

    if opts.print_cm : 
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random Forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%S penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty, dual=False, tol=1e-3)))
    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
])))


#########################################################################################


# TF-IDF is determined for words in the document 'corpus' 
# Dictionary is returned with word:value sorted in desending order by TF-IDF
# 'topicModel' is a boolean to determine if the topic model should be determined
# ..and returned during this instance
def build_tfidf(corpus, topicModel) :
    no_features = 100
    tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1,3), 
            max_features=no_features, 
            stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    idf = tfidf_vectorizer.idf_

    vectorDict = dict(zip(tfidf_feature_names, idf))
    sortedDict = sorted(vectorDict.items(), key=operator.itemgetter(1), reverse=True)

    if topicModel :
        display_topics(tfidf, tfidf_feature_names)

    return sortedDict


# Prints no_top_words for each feature using NMF algorithm
def display_topics(model, feature_names) :
    no_topics = 5
    no_top_words = 10
    nmf = NMF(n_components=no_topics, random_state=1).fit(model)
    for topic_idx, topic in enumerate(nmf.components_):
        print("Topic #%d:" %(topic_idx))
        print(" ,".join([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print()


# Store all reviews from a review_lst as a list 
def buildReviewLst(review_lst) :
    reviews = []
    for review in review_lst:
	if review in review_idx :
	    reviews.append(review_idx[review])
    return reviews


# Compact all reviews in a review_lst as a single string
def build_review_str(review_lst) :
    reviews = buildReviewLst(review_lst)
    reviewLine = ' '.join(reviews)
    return reviewLine

