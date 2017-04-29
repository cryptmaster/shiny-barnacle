# Reads the train/test files for tasks on YELP data
# ..creates '.key' files for ingest into score_rank_lst.py
# ..where final scoring is calculated

import glob, sys, os

task = sys.argv[1]

task_dir = "tasks/%s"%(task)
print 'Opening %s...'%(task_dir)
os.chdir(task_dir)
key_dat = {}
for file in glob.glob("*.train"):
    base = os.path.basename(file)
    basename = os.path.splitext(base)[0]
    test_file = '%s.test'%(basename)

    train_lst = []
    test_lst = []
    for line in open(file) :
        line = line.rstrip('\n')
        p = line.split(',')
        if int(p[1]) == 1 :
            l = 1
        else :
            l = -1
        newLine = p[0] + ' 5 ' + str(l) + ' TRAIN'
        train_lst.append(newLine)

    for line in open(test_file) :
        line = line.rstrip('\n')
        p = line.split(',')
        if int(p[1]) == 1 :
            l = 1
        else :
            l = -1
        newLine = p[0] + ' 5 ' + str(l) + ' TEST'
        test_lst.append(newLine)

    print 'Creating glob for %s with %d train, %d test, %d overall'%(
           basename, len(train_lst), len(test_lst), len(train_lst + test_lst))
    key_dat[basename] = train_lst + test_lst 


os.chdir(sys.path[0])
score_dir = 'projects/tasks/%s'%(task)
os.system('mkdir -p %s'%(score_dir));
here = os.path.dirname(os.path.realpath(__file__));
for tasktype in key_dat :
    statusfile = '%s/%s.key'%(score_dir,tasktype);
    print 'Writing to %s'%(statusfile)
    fid = open(statusfile,'w')
    fid.write("\n".join(key_dat[tasktype]))
    fid.close() 


print 'Making score results'
scorefiles = 'projects/scores/%s'%(sys.argv[2])
print'python score_rank_list.py -l %s -s %s'%(score_dir, scorefiles)
results = '%s/score_rank_list.results'%(scorefiles)
os.system('python score_rank_list.py -l %s -s %s > %s'%(score_dir, scorefiles, results))
print 'Completed -- saved to %s'%(results)
