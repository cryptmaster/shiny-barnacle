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
    print 'ERROR: 1 argument (test_cond) expected. 0 given'
    print '\tExpected arguments: categories | cities'
    sys.exit()

########################################
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
train_l = {}
test_l = {}
x_train = []
x_test = []
have_x = False
bid_lst = []
for file in glob.glob("*.train"):
    base = os.path.basename(file)
    basename = os.path.splitext(base)[0]
    test_file = '%s.test'%(basename)

    lst = []
    for line in open(file) :
        line = line.rstrip('\n')
        p = line.split(',')
        lst.append(p[1])
        if not have_x :
            x_train.append(''.join(review_txt[p[0]]))
    train_l[basename] = lst

    lst = []
    for line in open(test_file) :
        line = line.rstrip('\n')
        p = line.split(',')
        bid = p[0]
        if bid in review_txt :
            lst.append(p[1])
            if not have_x :
                x_test.append(''.join(review_txt[bid]))
                bid_lst.append(bid)
    test_l[basename] = lst
    have_x = True
printTime()


print 'Determining probability scores'
os.chdir(sys.path[0])
datetime = time.strftime("%Y%m%d_%H%M")
score_dir = 'projects/scores/TFIDF_%s_%s'%(test_cond,datetime)
os.system('mkdir -p %s'%(score_dir));
here = os.path.dirname(os.path.realpath(__file__));
for item in test_l:
    statusfile = '%s/Classification_Report.status'%(score_dir);
    fid = open(statusfile,'a')

    # Making conversion of negatives represented from 0 to -1 for sorting
    processingStatus = ('\n\nPreparing data for %s %s --------------\n'%(test_cond, item))
    print processingStatus
    fid.write(processingStatus)
    y_train = map(int, train_l[item])
    y_test = map(int, test_l[item])
    y_train = [x if x==1 else -1 for x in y_train]
    y_test = [x if x==1 else -1 for x in y_test]

    # Build over train
    print '\tBuilding over train...'
    pipe = make_pipeline(TfidfVectorizer(stop_words="english"), LogisticRegression())
    param_grid = {'logisticregression__C': [0.01, 1, 10, 100]}
#                  'tfidfvectorizer__ngram_range': [(1,1), (1,2)]} 
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(x_train, y_train)

    # Train status update
    bestScore = "Best cross-validation score: \t{:.2f}".format(grid.best_score_)
    bestParams = "Best parameters: \t{}".format(grid.best_params_)
    fid.write(bestScore + '\n')
    print '\t\t' + bestScore
    fid.write(bestParams + '\n')
    print '\t\t' + bestParams
    fid.write("\nGrid scores on development set:")
    for params, mean_score, scores in grid.grid_scores_:
        fid.write("\n\t%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() *2, params))

    # Evaluate over test
    print '\tEvaluating test...'
    testScore = 'Generalized performance assessment on test: {:.2f}'.format(grid.score(x_test, y_test))
    fid.write('\n\n' + testScore + '\n')
    print '\t\t' + testScore
    y_true, y_pred = y_test, grid.predict_proba(x_test)[:,1]
    fid.close()

    # Actual scores printed to .score files
    print '\tWriting .score file...'
    outfile = '%s/%s.scores'%(score_dir,item);
    fid = open(outfile,'w')
    fid.write('\n'.join(['%s %.6f %d'%(x[0],x[1],x[2]) for x in zip(bid_lst, y_pred, y_true)])+'\n')
    fid.close()
    printTime() 
    print '\n\n'
print '\nCompleted script runtime. Evaluated in: '
printTime() 


#print'python score_rank_list.py -l lists_%s/ -s %s'%(test_cond,score_dir);
#scorefile = '%s/score_rank_list.results'%(score_dir)
#os.system('python score_rank_list.py -l lists_%s/ -s %s > %s'%(test_cond,score_dir,scorefile));
