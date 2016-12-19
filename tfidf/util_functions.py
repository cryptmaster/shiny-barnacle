import numpy as np
import scipy.stats as ss
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import random

def read_key(keyfile,idx) :
    train_lst = [];
    test_lst = [];
    for l in open(keyfile) :
        p = l.split();
        if p[3] == 'TRAIN' : 
            train_lst.append([p[0],idx[p[0]],int(float(p[2]))]);
        else :
            test_lst.append([p[0],idx[p[0]],int(float(p[2]))]);
        #end
    #end
    return (train_lst,test_lst);
#end

def evaluate_scores(scores,lst,outfile=None) :
    score_rank = len(scores)-ss.rankdata(scores);
    lst_ids = [];
    lst_scores = [];
    lst_ranks = [];
    lst_labels = [];
    for (b,idx,l) in lst :
        lst_ids.append(b);
        lst_scores.append(scores[idx]);
        lst_ranks.append(score_rank[idx]);
        lst_labels.append(l);
    #end
    if outfile is not None :
        fid = open(outfile,'w');
        fid.write('\n'.join(['%s %.6f %.1f %d'%(x[0],x[1],x[2],x[3]) for x in zip(lst_ids,lst_scores,lst_ranks,lst_labels)])+'\n');
        fid.close();
    #end
    lst_scores = np.array(lst_scores);
    lst_labels = np.array(lst_labels);
    lst_ranks = np.array(lst_ranks);
    
    prior = float(np.sum(lst_labels==1))/len(lst_labels);
    ap = metrics.average_precision_score((lst_labels==1)+0,lst_scores+np.min(lst_scores));
    #[p,r,tap] = metrics.precision_recall_curve((lst_labels==1)+0,lst_scores+np.min(lst_scores));
    auc = metrics.roc_auc_score((lst_labels==1)+0,lst_scores+np.min(lst_scores));
    #[fp,tp,tauc] = metrics.roc_curve((lst_labels==1)+0,lst_scores+np.min(lst_scores));

    #for x in sorted(zip(lst_scores,lst_ranks,lst_labels), key = lambda x : x[1]) : print '%.4f %d %d'%(x); #end
    #for x in zip(p,r,fp,tp) : print '(%.2f,%.2f) (%.2f,%.2f)'%(x); #end
    
    mean_pos_rank = np.mean(lst_ranks[lst_labels==1]);
    mean_neg_rank = np.mean(lst_ranks[lst_labels<1]);
    mean_ratio = mean_neg_rank/mean_pos_rank;

    median_pos_rank = np.median(lst_ranks[lst_labels==1]);
    median_neg_rank = np.median(lst_ranks[lst_labels<1]);
    median_ratio = median_neg_rank/median_pos_rank;
    
    sl = zip(lst_scores,(lst_labels==1)+0);
    random.shuffle(sl);
    sl.sort(key=lambda x: x[0],reverse=True);
    
    tp = len([x for x in sl if x[1]==1]);
    tn = len([x for x in sl if x[1]==0]);
    t = len(sl);

    hits = 0;
    miss = np.zeros(len(sl)-1);
    fa = np.zeros(len(sl)-1);
    for k in range(len(sl)-1) :
        hits += sl[k][1];
        miss[k] = float(tp-hits)/tp;
        fa[k] = float((k+1)-hits)/tn;
    #end
    diff = np.abs(miss-fa);
    I = np.argmin(diff)
    eer = 0.5*(miss[I] + fa[I]);

    [rho,p] = ss.mstats.spearmanr(lst_scores,(lst_labels==1)+0);

    # Oracle calibration of scores
    #lr = LogisticRegression(class_weight='balanced');
    #lr.fit(lst_scores,(lst_labels==1)+0);
    #lst_scores_oc = lr.predict_proba(lst_scores)[:,1];
    #sl_oc = zip(lst_scores_oc,(lst_labels==1)+0);
    
    return (prior,ap,auc,mean_ratio,median_ratio,eer,rho,p,sl)
#end
