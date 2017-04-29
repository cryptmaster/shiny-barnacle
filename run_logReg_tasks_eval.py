# File: logReg_tasks_eval.py
# 
# This script has been modified from its original version to iterate over tasks
# ..relevant to the Yelp! dataset used. 
#
# Differing from the first edition, this script will classify our dataset using
# ..results from performing embed_symmetric_operator (Jeremy Silver) to identi-
# ..fy eingien values and context features with respect to the original ones(A)
# ..matrix
import json
import scipy.sparse as sp
import numpy as np
import pickle
import time
import util_functions as util
import sys, glob, os
import math
import operator
from numpy import genfromtxt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.append('/home/hltcoe/gsell/tools/python_mods/');
start = time.clock();

if len(sys.argv) == 2 :
    test_cond = sys.argv[1]
else :
    print'ERROR: 1 argument (results file) expected. 0 given'
    sys.exit()

# Print time elapse in seconds && minutes
def printTime() :
    timeElapse = time.clock()-start 
    print '\t%.2f seconds elapsed -- %.2f minutes'%(timeElapse, timeElapse/60)

#########################################################################
print "Loading vn_math.result from memory"
result_data = genfromtxt('vn_math.result', delimiter=',')
with open('bidID_lst.txt', "rb") as fp:
    bidID_lst = pickle.load(fp)
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
score_dir = 'projects/scores/TFIDF_%s'%(datetime)
os.system('mkdir -p %s'%(score_dir));
here = os.path.dirname(os.path.realpath(__file__));
for item in test_lst :

    print '\n\nProcessing %s --------------\n'%(item)
    # Making conversion of negatives represented from 0 to -1 for sorting
    y_train = map(int, y_train)
    y_test = map(int, y_test)
    y_train = [x if x==1 else -1 for x in y_train]
    y_test = [x if x==1 else -1 for x in y_test]

    # Build over train
    # **********
    # Here, we want to use the numbers determined previously as the classifier using
    # ..logistic regression. Then, using the train/test info for cities run the grid
    # ..classification process to assess performance and accuracy
    # **********
    print '\tBuilding over train...'
    pipe = make_pipeline(TfidfVectorizer(stop_words="english"), LogisticRegression())
    param_grid = {'logisticregression__C': [0.01, 1, 10, 100]}
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(text_train, y_train)
    print "Best cross-validation score: \t{:.2f}\n".format(grid.best_score_)
    print "Best parameters: \t{}\n".format(grid.best_params_)
    printTime()

    # Evaluate over test
    print '\tEvaluating test...'
    print '\n\nGeneralized performance assessment on test: {:.2f}\n'.format(grid.score(text_test, y_test))
    y_true, y_pred = y_test, grid.predict_proba(text_test)[:,1]
    printTime()

    # Actual scores printed to .score files
    print '\tWriting .score file...'
    outfile = '%s/%s.scores'%(score_dir,item);
    fid = open(outfile,'w')
    fid.write('\n'.join(['%s %.6f %d'%(x[0],x[1],x[2]) for x in zip(bid_lst, y_pred, y_true)])+'\n')
    fid.close()
    printTime() 
printTime() 
