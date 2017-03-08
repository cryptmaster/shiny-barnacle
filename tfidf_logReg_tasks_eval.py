# File: tfidf_logReg_tasks_eval.py
# 
# This script goes over the Yelp data stored at variable review_file to generate
# ..TF-IDF values for each document, where a document is defined as all reviews
# ..given to a business. TF-IDF is determined using feature_extraction from the 
# ..scikit-learn.org sklearn package (v0.16)
# 
# This script has been modified from its original version to iterate over tasks
# ..relevant to the Yelp! dataset used. Tasks are divded between 'categories' 
# ..and 'cities'.

import json
import scipy.sparse as sp
import numpy as np
import pickle
import time
import util_functions as util
import sys, glob, os
import math
import operator

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


sys.path.append('/home/hltcoe/gsell/tools/python_mods/');
business_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_business.json';
review_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_review.json';
start = time.clock();

if len(sys.argv) == 2 :
    test_cond = sys.argv[1]
else :
    print'ERROR: 1 argument (test_cond) expected. 0 given'
    print'\tExpected arguments: categories | cities'
    sys.exit()



# Print time elapse in seconds && minutes
def printTime() :
    timeElapse = time.clock()-start 
    print '\t%.2f seconds elapsed -- %.2f minutes'%(timeElapse, timeElapse/60)


print 'Loading reviews...'
review_str = [x for x in open(review_file).read().split('\n') if len(x)>0];
review_txt = {}
rid_txt = {}
for l in review_str :
    obj = json.loads(l);
    business = obj['business_id']
    rid = obj['review_id']
    if business not in review_txt :
        review_txt[business] = []
    review_txt[business].append(obj['text'])
    rid_txt[rid] = obj['text']
printTime()


print 'Generating train/test lists for %s task...'%(test_cond)
task_dir = "tasks/%s"%(test_cond)
os.chdir(task_dir)
train_lst = {}
test_lst = {}
for file in glob.glob("*.train"):
    base = os.path.basename(file)
    basename = os.path.splitext(base)[0]
    test_file = '%s.test'%(basename)
    lst = []
    for line in open(file) :
        line = line.rstrip('\n')
        p = line.split(',')
        lst.append([p[0],p[1]])
    train_lst[basename] = lst
    lst = []
    for line in open(test_file) :
        line = line.rstrip('\n')
        p = line.split(',')
        lst.append([p[0],p[1]])
    test_lst[basename] = lst
printTime()


print 'Determining probability scores'
os.chdir(sys.path[0])
datetime = time.strftime("%Y%m%d_%H%M")
score_dir = 'projects/scores/TFIDF_%s_%s'%(test_cond,datetime)
os.system('mkdir -p %s'%(score_dir));
here = os.path.dirname(os.path.realpath(__file__));
for item in test_lst :
    statusfile = '%s/Classification_Report.status'%(score_dir);
    fid = open(statusfile,'a')

    processingStatus = ('\n\nProcessing %s %s --------------\n'%(test_cond, item))
    print processingStatus
    fid.write(processingStatus)
    text_train = []
    y_train = []
    for (b,l) in train_lst[item] :
        if b in review_txt :
            text_train.append(''.join(review_txt[b]))
            y_train.append(l)
        else :
            print b + ' not in review_txt'
    text_test = []
    y_test = []
    bid_lst = []
    missing = []
    for (b,l) in test_lst[item] :
        if b in review_txt :
            bid_lst.append(b)
            text_test.append(''.join(review_txt[b]))
            y_test.append(l)
        else :
            missing.append(b)

    # Making conversion of negatives represented from 0 to -1 for sorting
    y_train = map(int, y_train)
    y_test = map(int, y_test)
    y_train = [x if x==1 else -1 for x in y_train]
    y_test = [x if x==1 else -1 for x in y_test]

    print 'FYI - test has %d found and %d missing from the total %d'%(len(y_test), len(missing), len(test_lst[item]))
    # Build over train
    print '\tBuilding over train...'
    pipe = make_pipeline(TfidfVectorizer(stop_words="english"), LogisticRegression())
    param_grid = {'logisticregression__C': [0.01, 1, 10, 100]}
#                  'tfidfvectorizer__ngram_range': [(1,1), (1,2)]} 
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(text_train, y_train)
    printTime()

    # Train status update
    bestScore = "Best cross-validation score: \t{:.2f}\n".format(grid.best_score_)
    bestParams = "Best parameters: \t{}\n".format(grid.best_params_)
    fid.write(bestScore)
    print bestScore
    fid.write(bestParams)
    print bestParams
    fid.write("\nGrid scores on development set:")
    for params, mean_score, scores in grid.grid_scores_:
        fid.write("\n\t%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() *2, params))

    # Evaluate over test
    print '\tEvaluating test...'
    testScore = '\n\nGeneralized performance assessment on test: {:.2f}\n'.format(grid.score(text_test, y_test))
    fid.write(testScore)
    print testScore
    y_true, y_pred = y_test, grid.predict_proba(text_test)[:,1]
    fid.write('Test values not found in JSON:')
    fid.write(', '.join(missing))
    fid.write('\n')
    fid.close()
    printTime()

    # Actual scores printed to .score files
    print '\tWriting .score file...'
    outfile = '%s/%s.scores'%(score_dir,item);
    fid = open(outfile,'w')
    fid.write('\n'.join(['%s %.6f %d'%(x[0],x[1],x[2]) for x in zip(bid_lst, y_pred, y_true)])+'\n')
    fid.close()
    printTime() 
printTime() 


#print'python score_rank_list.py -l lists_%s/ -s %s'%(test_cond,score_dir);
#scorefile = '%s/score_rank_list.results'%(score_dir)
#os.system('python score_rank_list.py -l lists_%s/ -s %s > %s'%(test_cond,score_dir,scorefile));
