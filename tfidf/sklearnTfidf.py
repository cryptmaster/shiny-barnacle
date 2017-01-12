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
import sklearn.metrics as metrics
sys.path.append('/home/hltcoe/gsell/tools/python_mods/');
review_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_review.json';
DEFAULT_IDF_UNITTEST = 1.0
test_cond = '12_45'
start = time.clock();
edge_type = sys.argv[1]

def printTime() :
    timeElapse = time.clock()-start 
    print '\t%.2f seconds elapsed -- %.2f minutes'%(timeElapse, timeElapse/60)


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
    for s in [0,1] :
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
		    d[0].append(reviewCounter)		# d
                    r[0].append(reviewer_idx[uid]) 	# r
                    c[0].append(business_idx[bid]) 	# c
    A[1] = sp.csr_matrix((d[1],(c[1],r[1])),shape=[B,R])
    A[0] = sp.csr_matrix((d[0],(c[0],r[0])),shape=[B,R])
    printTime()
    return A


def buildVector(corpus, posDic, negDic) :
#    vectorizer = text.CountVectorizer(input='content',stop_words='english')
#    dtm = vectorizer.fit_transform(reviews).toarray()
#    vocab = np.array(vectorizer.get_feature_names())    print dtm.shape
#    print len(vocab)
    userRating = 0
    totalRating = 0
    if len(corpus) > 0 :
        vectorizer = TfidfVectorizer(
            smooth_idf=False,
            min_df=1, max_df=1.0, max_features=None,
            stop_words='english', 
            )
        X = vectorizer.fit_transform(corpus)
        idf = vectorizer.idf_
        vectorDict = dict(zip(vectorizer.get_feature_names(), idf))
        sortedDict = sorted(vectorDict.items(), key=operator.itemgetter(1), reverse=True)

        for (word,value) in sortedDict :
            totalRating += value
            if word in posDic :
                userRating += value
            if word in negDic :
                userRating -= value 
    userValue = 0
    if userRating > 0 :
        userValue = float(userRating)/totalRating
    return userValue


# Create a dictionary of words 
def buildDictionary(review) :
    dictionary = []
    if len(review) > 0 :
        vectorizer = TfidfVectorizer(
            smooth_idf=False,
            min_df=1, max_df=1.0, max_features=None,
            stop_words='english', 
            )
        X = vectorizer.fit_transform(review)
        idf = vectorizer.idf_
        vectorDict = dict(zip(vectorizer.get_feature_names(), idf))
        vectorDict = sorted(vectorDict.items(), key=operator.itemgetter(1), reverse=True)
        for word in vectorDict[:10] :
            dictionary.append(word[0])
    return dictionary 


# Store all reviews from review list as a list
def buildReviewLst(review_lst) :
    reviews = []
    for review in review_lst:
	if review in review_idx :
	    reviews.append(review_idx[review])
    return reviews


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
        posDic = []
        negDic = []
        pos_reviews = []
        neg_reviews = []
        for (b,i,l) in train_lst :
            if l == 1 :
                pos_reviews = buildReviewLst(A[1].getrow(business_idx[b]).tocoo().data)
            else :
                neg_reviews = buildReviewLst(A[0].getrow(business_idx[b]).tocoo().data)
            posDic += buildDictionary(pos_reviews)
            negDic += buildDictionary(neg_reviews)

        posDic = remove_duplicates(posDic)
        negDic = remove_duplicates(negDic)
#        print 'Positive Dictionary: '
#        print posDic
#        print '\n\nNegative Dictionary:'
#        print negDic

        confidence = 0
        for (b,i,l) in test_lst :
            for s in [0,1] :
                train_reviews = A[s].getrow(business_idx[b]).tocoo().data
            reviews = buildReviewLst(train_reviews)
            userRating = float(buildVector(reviews, posDic, negDic))*100
            if userRating > .75 :
                label = 1
            else :
                label = -1

            if str(label) == str(l) :
                confidence += 1
        accuracy = (float(confidence)/len(test_lst))*100
        print "\n\nOVERALL STATS FOR REVIEWER: " + str(reviewer)
        print "\tPositive Dictionary: \t"+ str(len(posDic))
        print "\tNegative Dictionary: \t"+ str(len(negDic))
        print "\tBusinesses trained: \t" + str(len(train_lst))
        print "\tBusinesses tested: \t" + str(len(test_lst))
        print "\t# Accurate ratings:\t%d"%(confidence)
        print "\t\tAccuracy Rating: %.2f%%"%(accuracy) 



# Obtain keywords for reviewer TFIDF and publish to .score files
def determineScores() :
    print 'Running evaluation...';
    score_dir = 'scores/TFIDF_%s'%(test_cond);
    os.system('mkdir -p %s'%(score_dir));
    os.system('rm %s/*'%(score_dir));
    here = os.path.dirname(os.path.realpath(__file__));
    revCount = 0

    for reviewer in test_reviewer_lst :
        outfile = '%s/%s.scores'%(score_dir,reviewer);
        revCount += 1
        [train_lst,test_lst] = util.read_key('lists_%s/%s.key'%(test_cond,reviewer),business_idx);
        rTFIDF = tfidf.TfIdf("tfidf_teststopwords.txt")	# TFIDF instance for each reviewer
        for (b,i,l) in test_lst :
            train_reviews = A.getrow(business_idx[b]).tocoo().data
            rDoc = ''
            for review in train_reviews :
                if review in review_idx :
                    rDoc += review_idx[review]
            rTFIDF.add_input_str(rDoc)
        printScores(rTFIDF, outfile)
        print "\nWROTE TO FILE %s ---- %d%% COMPLETE"%(outfile, (float(revCount)/len(test_reviewer_lst))*100)
    printTime()


def printScores(TFIDF, outfile) :
    fid = open(outfile,'w');
    tokens = TFIDF.return_tokens()
    keywords = TFIDF.get_doc_keywords()
    for word in keywords[:10] : 
	fid.write("\n\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens[word[0]],TFIDF.get_idf(word[0]),word[1]))
    fid.write("\n")
    fid.close()
    printTime()

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
#determineScores()


# EOF
