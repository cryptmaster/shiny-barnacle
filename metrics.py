#!/usr/local/bin/python

import sys,os,re,string,random
import sklearn.metrics as skm
import scipy.stats as ss
import numpy as np

def computeall(hits, Ptgt = 0.1, Cmiss = 1, Cfa = 1):

    hits.sort(key = lambda x: x[0], reverse=True)

    # Compute the EER

    errors = []
    TP = 0.
    FP = 0.
    totTP = sum([h[3] for h in hits if h[1] == 1])
    totFP = sum([h[4] for h in hits if h[1] == 0])

    Npos = 0
    Nneg = 0
    for k in range(len(hits)):
        Npos += int(hits[k][1])
        Nneg += int(not hits[k][1])
        TP += hits[k][3]*float(hits[k][1] == 1)
        FP += hits[k][4]*float(hits[k][1] == 0)
        errors.append((float(FP)/totFP,1.0-float(TP)/totTP))

    diff = [ abs(x[0]-x[1]) for x in errors ]
    I = np.argmin(diff)
    EER = 0.5*(errors[I][0] + errors[I][1])

    # Compute the EER stddev
    stdEER = 0.5*np.sqrt((EER+1.0/(Npos+Nneg))*(1-(EER+1./(Npos+Nneg)))*(1.0/Npos + 1.0/Nneg))

    # Compute average precision
    ap = skm.average_precision_score([x[1] for x in hits],[x[0] for x in hits]);

    # Compute ROC AUC
    roc = skm.roc_auc_score([x[1] for x in hits],[x[0] for x in hits]);

    # Compute spearman's rho
    [rho,p] = ss.spearmanr([x[1] for x in hits],[x[0] for x in hits]);

    rmse = np.sqrt(np.mean(np.array([(x[0]-x[5])**2 for x in hits])));

    return ((Npos,Nneg,ap,roc,rho,EER,rmse,p,stdEER),errors)
