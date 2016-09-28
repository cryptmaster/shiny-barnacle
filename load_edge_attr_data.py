import json
import scipy.sparse as sp
import numpy as np
import pickle
import time

import sys

start = time.clock();

business_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_business.json';
review_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_review.json';

print 'Loading reviewer lists...'
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Loading businesses...'
business_str = [x for x in open(business_file).read().split('\n') if len(x)>0];
bid_info = {};
for l in business_str :
    obj = json.loads(l);
    bid_info[obj['business_id']] = obj;
#end
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Loading reviewers and reviews...'
test_reviewer_lst = [x for x in open('lists_%s/test_reviewers.lst'%(test_cond)).read().split('\n') if len(x)>0]
review_str = [x for x in open(review_file).read().split('\n') if len(x)>0];
reviewer_lst = set([]);
review_info = {};
total_review_cnt = 0;
for l in review_str :
    obj = json.loads(l);
    reviewer_lst.add(obj['user_id']);
    review_info[obj['review_id']] = obj;
#end
train_reviewer_lst = sorted(list(reviewer_lst-set(test_reviewer_lst)));
reviewer_lst = sorted(list(reviewer_lst));
review_lst = sorted(review_info.keys());

reviewer_reviews = {};
for r in train_reviewer_lst+test_reviewer_lst : reviewer_reviews[r] = []; #end
for r in review_lst :
    reviewer_reviews[review_info[r]['user_id']].append(r);
#end
print '   %.2f seconds elapsed'%(time.clock()-start);


print 'Linking reviewers to reviewed businesses...'
bid_reviewers = {};
bid_reviews = {};
bid_lst = sorted(bid_info.keys());
for b in bid_lst :
    bid_reviewers[b] = [];
    bid_reviews[b] = [];
#end
for r in train_reviewer_lst :
    for rr in reviewer_reviews[r] :
        bid_reviewers[review_info[rr]['business_id']].append(r);
        bid_reviews[obj['business_id']].append(rr);
    #end
#end
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Creating reviewer index lookup...'
train_reviewer_ndx = {};
cnt = 0;
for r in train_reviewer_lst :
    train_reviewer_ndx[r] = cnt;
    cnt += 1;
#end
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Removing businesses with no reviews...'
reviewed_bid_lst = [];
for b in bid_lst :
    if len(bid_reviewers[b]) > 0 : reviewed_bid_lst.append(b); #end
#end
reviewed_bid_lst.sort();
print '   Reduced from %d to %d'%(len(bid_lst),len(reviewed_bid_lst));
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Loading into data dictionary...'
data = {};
data['Train Reviewer List'] = train_reviewer_lst;
data['Test Reviewer List'] = test_reviewer_lst;
data['Business Information'] = bid_info;
data['Review Information'] = review_info;
data['Reviewer Reviews'] = reviewer_reviews;
data['Reviewed Business List'] = reviewed_bid_lst;
#pickle.dump(to_save,open('projects/pickle/edge_attr_task_data.pkl','w'));
print '   %.2f seconds elapsed'%(time.clock()-start);

"""
print 'Collecting metadata lists...'
city = [];
single_categories = [];
grouped_categories = [];
for b in id_info.keys() :
    city.append(id_info[b]['city']+','+id_info[b]['state']);
    grouped_categories.append(tuple(id_info[b]['categories']));
    single_categories += id_info[b]['categories'];
#end
city_lst = sorted(list(set(city)));
print '   %d cities'%(len(city_lst));
single_category_lst = sorted(list(set(single_categories)));
print '   %d single categories'%(len(single_category_lst));
grouped_category_lst = sorted(list(set(grouped_categories)));
print '   %d groups of categories'%(len(grouped_category_lst));
print '   %.2f seconds elapsed'%(time.clock()-start);
"""

"""
print 'Printing business table...'
city_ndx = {};
for n in range(len(city_lst)) :
    city_ndx[city_lst[n]] = n;
#end
gc_ndx = {};
for n in range(len(grouped_category_lst)) :
    gc_ndx[grouped_category_lst[n]] = n;
#end

cnt = 0;
table_str = 'vertex_ndx|business_id|business_name|review_count|city_ndx|city|group_category_ndx|group_category\n';
fid = open('business_info.tbl','w');
for b in reviewed_id_lst :
    table_str += '%d|%s|%s|'%(cnt,b.encode('utf-8'),id_info[b]['name'].encode('utf-8'));
    table_str += '%d|'%(id_info[b]['review_count']);
    table_str += '%d|%s|'%(city_ndx[id_info[b]['city']+','+id_info[b]['state']],id_info[b]['city'].encode('utf-8')+','+id_info[b]['state'].encode('utf-8'));
    table_str += '%d|%s\n'%(gc_ndx[tuple(id_info[b]['categories'])],','.join(id_info[b]['categories']).encode('utf-8'));
    cnt += 1;
    if cnt%1000 == 0 :
        fid.write(table_str);
        table_str = '';
    #end
#end
fid.write(table_str);
fid.close();
print '   %.2f seconds elapsed'%(time.clock()-start);


print 'Building Business x Reviewer index...'
r = [];
c = [];
for n in range(len(reviewed_bid_lst)) :
    for u in bid_reviewers[reviewed_bid_lst[n]] :
        r.append(n);
        c.append(reviewer_ndx[u]);
    #end
#end
A = sp.csr_matrix((np.ones((len(r),)),(r,c)));
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Building graph from common reviewers...'
K = A.dot(A.T);
print '   %.2f seconds elapsed'%(time.clock()-start);

print 'Printing edge list...'
Kcoo = sp.coo_matrix(K);
r = Kcoo.row;
c = Kcoo.col;
d = Kcoo.data;
print_str = '';
fid = open('business_review_edges.lst','w');
for n in range(len(r)) :
    if r[n] <= c[n] :
        print_str += '%d %d %d\n'%(r[n],c[n],d[n]);
        if n%1000 == 0 :
            print n,len(r)
            fid.write(print_str);
            print_str = '';
        #end
    #end
#end
fid.write(print_str);
fid.close();

print 'Analyzing...'
K1 = sp.csr_matrix((np.ones_like(K.data),K.indices,K.indptr));
[Ncc,CC] = sp.csgraph.connected_components(K1,directed=False);
print '   %d connected components'%(Ncc);
ccMax = 0;
for n in range(Ncc) :
    ccMax = np.maximum(np.sum(CC==n),ccMax);
#end
print '   Largest component: %d'%(ccMax);
D = np.array(K1.sum(axis=0));
maxD = int(np.max(D));
cnt = np.zeros((maxD,));
for n in range(maxD) :
    cnt[n] = np.sum(D==n);
#end
print '   %d Singletons'%(cnt[1]);
print '   %d Spokes'%(cnt[2]);
print '   %d Maximum Degree'%(maxD);
print '   %.2f seconds elapsed'%(time.clock()-start);
"""
