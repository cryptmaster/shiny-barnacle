#!/export/apps/bin/python

import sys,os,re,string,random,metrics
import getopt
import numpy as np
import random

from copy import deepcopy

def perfstats( users, userhits, pooled=0 ):
    if len(users) == 0:
        return []

    det = []
    userperf = {}
    if pooled:
        allhits = []
        for u in users:
            allhits += userhits[u]
        userperf['ALL'] = metrics.computeall(allhits)
        users = ['ALL']
        det = userperf['ALL'][1]
    else:
        for u in users:
            userperf[u] = metrics.computeall(userhits[u])

    dim = len(userperf[users[0]][0])-4
    fos = np.zeros(dim)
    sos = np.zeros(dim)
    for u in users:
        for d in range(dim) :
            fos[d] += np.array(userperf[u][0][2+d])
            sos[d] += np.array(userperf[u][0][2+d])**2

    meanp = fos/len(users)
    std = np.sqrt(sos/len(users)- meanp**2)

    if len(users) == 1:
        std[-1] = userperf[users[0]][0][-1]
        std[-2] = userperf[users[0]][0][-2]
        npos = userperf[users[0]][0][0]
        nneg = userperf[users[0]][0][1]
    else:
        std[-1] = 0
        npos = 0
        nneg = 0
        for u in users:
            std[-1] += userperf[u][0][-1]**2
            std[-2] = userperf[users[0]][0][-2]
            npos += userperf[u][0][0]
            nneg += userperf[u][0][1]
        std[-1] = np.sqrt(std[-1])/len(users)
        std[-2] = np.sqrt(std[-2])/len(users)

    return (meanp.tolist(),std.tolist(),npos,nneg,det)

def printstats( desc, stats, nusers, resfid, sepline, pool=0 ):    
    if len(stats) == 0:
        return

    if nusers > 1:
        if pool == 1:
            desc += " (WP)"
    
    mean = stats[0]
    var = stats[1]
    npos = stats[2]
    nneg = stats[3]

    outstr = "| " + desc.ljust(30)[:30] + " | " + str(nusers).rjust(2) \
        + " | " + str(npos).rjust(4) + " | " + str(nneg).rjust(4) + " | "
    for n in range(len(mean)):
        s = ("%0.3f" % mean[n])[:5] + " +\- " + ("%0.3f" % var[n])[:5]
        outstr += s.rjust(15) + " | "

    outstr = outstr.strip()
    print >>resfid, outstr
    print >>resfid, sepline

    return outstr.strip()


def oracle_calibration(hits):
    import sklearn.linear_model.logistic as logit
    model = logit.LogisticRegression(class_weight="auto",C=1.0,penalty='l2',tol=0.000001)

    x = np.zeros((len(hits),1))
    kx = np.zeros((len(hits),4))
    y = np.zeros((len(hits),))

    for n in range(len(hits)):
        #x[n] = np.log(hits[n][0]/(1-hits[n][0]+1e-10))
        x[n] = hits[n][0]
        kx[n][0] = x[n]
        kx[n][1] = x[n]**2
        kx[n][2] = x[n]**3
        kx[n][3] = x[n]**4
        y[n] = hits[n][1]

    model.fit(kx,y)
    ocy = model.predict_proba(kx)
    for n in range(len(hits)):
        hits[n] = (ocy[n][1].tolist(),hits[n][1],hits[n][2])

    return hits


def usage():
    print '\nscore_rank_list.py'
    print '\t -s: sff directory'
    print '\t -l: list directory'
    print '\t -e: evaluation set (dev or test)'
    print '\t -o: output results file ("-" for stdout)'
    print '\t -p: pooled metrics'
    print '\t -c: oracle calibration'

def main(argv):
    try:
        [opts,args] = getopt.getopt(argv,'hl:s:l:e:o:pl:cl:d:');
    except getopt.GetoptError as err:
        print str(err)
        usage();
        sys.exit(1);

    evalset = "dev"
    labeldir = ""
    scoredir = ""
    resfile = "-"
    pooled = False
    oracle = False
    detfile = ""

    for opt, arg in opts:
        if opt == "-e":
            evalset = arg.lower()
        elif opt == "-l":
            labeldir = arg;
        elif opt == "-s":
            scoredir = arg;
        elif opt == "-o":
            resfile = arg;
        elif opt == "-d":
            detfile = arg;
        elif opt == "-p":
            pooled = True
        elif opt == "-c":
            oracle = True
        elif opt == "-h":
            usage()
            sys.exit(0);

    if resfile == "-":
        resfid = sys.stdout
    else:
        resfid = open(resfile,'w')


    labelfiles = os.listdir(labeldir);
    labelfiles = filter(lambda x: ".key" in x, labelfiles)
    userlist = [k.strip().split('.')[0] for k in labelfiles]

    userhits = {}
    useroraclehits = {}
    userpos = {}
    userneg = {}

    # Accumulate trial scores for each user
    for user in userlist:

        # Read in eval set labels
        labelfile = labeldir+'/'+user+'.'+evalset
        labels = {}
        userpos[user] = 0
        userneg[user] = 0

        # Read in score file
        scorefile = scoredir+"/"+user+".scores"
        names = [];
        scores = [];
        labels = [];
        for l in open(scorefile) :
            x = l.strip().split();
            score = float(x[1]);
            label = int(float(x[-1]));
            if label == 1 :
                userpos[user] += 1;
            else :
                userneg[user] += 1;
                label = 0;
            #end
            names.append(x[0]);
            scores.append(score);
            labels.append(label);
        #end

        #Get hit list for the user
        hits = []
        for n in range(len(scores)) :
            s = scores[n]+1e-10*np.random.random();
            newhit = (s,labels[n],names[n])
            hits.append(newhit)

        # Apply oracle calibration if requested
        oracle_hits = oracle_calibration(hits)            

        # If pooled, set user prevalence weights
        userwtpos = 1
        userwtneg = 1
        if pooled:
            userwtpos = 1./userpos[user]
            userwtneg = 1./userneg[user]

        userhits[user] = [(h[0],h[1],h[2],userwtpos,userwtneg) for h in hits]        
        useroraclehits[user] = [(h[0],h[1],h[2],userwtpos,userwtneg) for h in oracle_hits]        
    #end

    # Print the table header
    outstr = "| " + "USER SET".ljust(30) + " |  N  | Npos | Nneg | " + "AP".center(15)+" |";
    outstr += "AUC".center(15)+" |"+"SPEARMAN".center(15)+" |"+"EER".center(15) + " |"
    dashes = "".ljust(len(outstr),"-")
    print >>resfid, dashes
    print >>resfid, outstr
    print >>resfid, dashes

    # Individual users
    for u in sorted(userlist,key=str.lower):
        usersub = [u]
        P = perfstats(usersub,userhits)
        printstats(u, P, len(usersub), resfid, dashes, pooled)

    # Overall results
    P = perfstats(userlist,userhits,False)
    printstats("Overall", P, len(userlist), resfid, dashes, False)
    P = perfstats(userlist,userhits,True)
    printstats("Overall", P, len(userlist), resfid, dashes, True)
    P = perfstats(userlist,useroraclehits,True)
    printstats("Overall", P, len(userlist), resfid, dashes, True)
    if pooled:
        if len(detfile) > 0:
            detfid = open(detfile,'w')
            for p in P[-1]:
                print >>detfid, str(p[0]) + " " + str(p[1])


if __name__ == '__main__':
    main(sys.argv[1:]);



        
