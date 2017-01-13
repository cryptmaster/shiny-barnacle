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
from sklearn.decomposition import NMF
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


# Magic
def buildTFIDF(corpus) :
    no_topics = 20
    no_features = 1000
    no_top_words = 10
    if len(corpus) > 0 :
        tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, max_features=no_features, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(corpus)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        idf = tfidf_vectorizer.idf_
        vectorDict = dict(zip(tfidf_vectorizer.get_feature_names(), idf))
        sortedDict = sorted(vectorDict.items(), key=operator.itemgetter(1), reverse=True)
	display_topics(nmf, tfidf_feature_names, no_top_words)
    return sortedDict


# Prints no_top_words for each feature
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" %(topic_idx)
        print " ".join([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]])


# Creates TFIDF weighted list of verbiage
def buildVector(corpus, posDic, negDic) :
    sortedDict = buildTFIDF(corpus)

    userRating = 0
    totalRating = 0
    for (word,value) in sortedDict :
        totalRating += value
        if word in posDic : userRating += value
        if word in negDic : userRating -= value 

    userValue = 0
    if userRating > 0 : userValue = float(userRating)/totalRating
    return userValue


# Create a dictionary of words from review
def buildDictionary(review) :
    sortedDict = buildTFIDF(review)

    dictionary = []
    for word in sortedDict[:10] : dictionary.append(word[0])

    return dictionary 


# Store all reviews from review list as a list
def buildReviewLst(review_lst) :
    reviews = []
    for review in review_lst:
	if review in review_idx :
	    reviews.append(review_idx[review])
    return reviews


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
    return output


# use Train data and Test data for each reviewer
def trainTest() :
    for reviewer in test_reviewer_lst :
        [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,reviewer),business_idx);
        # Build positive and negative dictionary based on 
        # ..reviewer's past data
        posDic = []
        negDic = []
        pos_reviews = []
        neg_reviews = []
        for (b,i,l) in train_lst :
            if l == 1 :
                pos_reviews = buildReviewLst(A[1].getrow(business_idx[b]).tocoo().data)
            else :
                neg_reviews = buildReviewLst(A[-1].getrow(business_idx[b]).tocoo().data)
            posDic += buildDictionary(pos_reviews)
            negDic += buildDictionary(neg_reviews)

        # Ensure dictionaries are free of duplicates
        posDic = remove_duplicates(posDic)
        negDic = remove_duplicates(negDic)
        for word in negDic[:] :
            if word in posDic :
                posDic.remove(word)
                negDic.remove(word)

        # Determine 'confidence' of predictive label based on presence
        # ...of business review's words in posDic && negDic
        confidence = 0
        for (b,i,l) in test_lst :
            for s in [-1,1] :
                train_reviews = A[s].getrow(business_idx[b]).tocoo().data
            reviews = buildReviewLst(train_reviews)
            userRating = float(buildVector(reviews, posDic, negDic))*100
            if userRating > .75 :
                label = 1
            else :
                label = -1

            if str(label) == str(l) :
                confidence += 1

        # Provide summary of intel
        accuracy = (float(confidence)/len(test_lst))*100
        print "\n\nOVERALL STATS FOR REVIEWER: " + str(reviewer)
        print "\tPositive Dictionary: \t"+ str(len(posDic))
        print "\tNegative Dictionary: \t"+ str(len(negDic))
        print "\n\tBusinesses trained: \t" + str(len(train_lst))
        print "\tBusinesses tested: \t" + str(len(test_lst))
        print "\n\tAccurate predictions:\t%d"%(confidence)
        print "\tAccuracy Rating: \t%.2f%%"%(accuracy) 
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
A = initialBuildIndex() 
trainTest()

# EOF
