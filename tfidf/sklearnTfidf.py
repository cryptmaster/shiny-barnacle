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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
from sklearn import linear_model, neighbors
import sklearn.metrics as metrics

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
def initialBuildIndex() :
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


# Bag-of-word example from Introduction to Machine Learning
# ..with Python by O'Reily
def bagOfWords(X_train, y_train, X_test, y_test) :
#    vect = CountVectorizer().fit(X_train)
#    X_train = vect.transform(X_train)
#    print(repr(X_train))

    no_features = 100
    vect = TfidfVectorizer(ngram_range=(1,3), max_features=no_features, stop_words='english')
    X_train = vect.fit_transform(X_train)
    print(repr(X_train))

    feature_names = vect.get_feature_names()
    print(len(feature_names))
    # print first 20 features
    print(feature_names[:20])
    # get every 20th word to get an overview
    print(feature_names[::20])

    scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
    scoreMean = np.mean(scores)
    print "Mean Score: " + str(scoreMean)

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print ("Best cross-validation score: ", grid.best_score_)
    print ("Best parameters: ", grid.best_params_)

    X_test = vect.transform(X_test)
    testScore = grid.score(X_test, y_test)
    print "Test score: " + str(testScore)


# use Train data and Test data for each reviewer
def trainTest() :
    for reviewer in test_reviewer_lst :
        [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,reviewer),business_idx);
        # Build positive and negative dictionary based on 
        # ..reviewer's past data
        print '\n\nPROCESSING FOR REVIEWER %s'%(str(reviewer))
        posDic = []
        negDic = []
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for (b,i,l) in train_lst :
            reviews = buildReviewLine(A[l].getrow(business_idx[b]).tocoo().data)
            X_train.append(reviews)
            y_train.append(l)
	print '\tcompleted train_lst import'

        for (b,i,l) in test_lst :
            reviews = []
            for s in [-1,1] :
                reviews.append(buildReviewLine(A[s].getrow(business_idx[b]).tocoo().data))
            reviews = ' '.join(reviews)
            X_test.append(reviews)
            y_test.append(l)
        print '\tcompleted test_lst import'

        bagOfWords(X_train, y_train, X_test, y_test)
        printTime()


# Magic
def buildTFIDF(corpus, topicModel) :
    no_features = 100
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    idf = tfidf_vectorizer.idf_
    vectorDict = dict(zip(tfidf_feature_names, idf))
    sortedDict = sorted(vectorDict.items(), key=operator.itemgetter(1), reverse=True)
    if topicModel :
        display_topics(tfidf, tfidf_feature_names)

    return sortedDict


# Prints no_top_words for each feature
def display_topics(model, feature_names) :
    no_topics = 5
    no_top_words = 10
    nmf = NMF(n_components=no_topics, random_state=1).fit(model)
    for topic_idx, topic in enumerate(nmf.components_):
        print "Topic #%d:" %(topic_idx)
        print " ,".join([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]])
        print ''


# Create a dictionary of words from review
def buildDictionary(review) :
    dictionary = []
    if len(review) > 0 :
        sortedDict = buildTFIDF(review, False)
        for word in sortedDict : dictionary.append(word[0])
    return dictionary 


# Store all reviews from a review_lst as a list of the string reviews
def buildReviewLst(review_lst) :
    reviews = []
    for review in review_lst:
	if review in review_idx :
	    reviews.append(review_idx[review])
    return reviews


# Compact all reviews in a review_lst as a single string
def buildReviewLine(review_lst) :
    reviews = buildReviewLst(review_lst)
    reviewLine = ' '.join(reviews)
    return reviewLine


# Eliminates duplicate words from pos & neg dictionaries
def remove_duplicates(values) :
    output = []
    seen = set()
    for value in values :
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen :
            output.append(value)
            seen.add(value)
    output = buildTFIDF(output, True)
    return output


# Creates TFIDF weighted list of verbiage
def buildVector(corpus, posDic, negDic) :
    userValue = 0
    if len(corpus) > 0 :
        sortedDict = buildTFIDF(corpus, False)

        userRating = 0
        totalRating = 0
        for (word,value) in sortedDict :
            totalRating += value
            if word in posDic : userRating += value
            if word in negDic : userRating -= value 
        if userRating > 0 : userValue = float(userRating)/totalRating
    return userValue



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
A = initialBuildIndex() 
trainTest()

# EOF
