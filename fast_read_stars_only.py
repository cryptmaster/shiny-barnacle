import scipy.sparse as sp
import time

st = time.clock();

test_cond='1_5';

print 'Loading lists...'
business_lst = [l for l in open('projects/csv/business_vertices_%s.csv'%(test_cond)).read().split('\n') if len(l)>0];
all_reviewer_lst = [l for l in open('projects/csv/reviewer_vertices_%s.csv'%(test_cond)).read().split('\n') if len(l)>0];
# Read business indices
business_idx = {};
for n in range(len(business_lst)) : business_idx[business_lst[n]] = n; #end
reviewer_idx = {};
for n in range(len(all_reviewer_lst)) : reviewer_idx[all_reviewer_lst[n]] = n; #end
print '   %.2f seconds elapsed'%(time.clock()-st);

print 'Loading train edges...'
train_edges = [x.split(',') for x in open('projects/csv/bipartite_train_edges_%s.csv'%(test_cond)).read().split('\n') if len(x)>0 and 'reviewer' not in x];
r_train = [int(float(x[0])) for x in train_edges];
c_train = [int(float(x[1])) for x in train_edges];
d_train = [int(float(x[2])) for x in train_edges];
print '   %.2f seconds elapsed'%(time.clock()-st);

print 'Loading seed edges...'
seed_edges = [x.split(',') for x in open('projects/csv/bipartite_seed_edges_%s.csv'%(test_cond)).read().split('\n') if len(x)>0 and 'reviewer' not in x];
r_seed = [int(float(x[0])) for x in seed_edges];
c_seed = [int(float(x[1])) for x in seed_edges];
d_seed = [int(float(x[2])) for x in seed_edges];
print '   %.2f seconds elapsed'%(time.clock()-st);

print 'Loading test edges...'
test_edges = [x.split(',') for x in open('projects/csv/bipartite_test_edges_%s.csv'%(test_cond)).read().split('\n') if len(x)>0 and 'reviewer' not in x];
r_test = [int(float(x[0])) for x in test_edges];
c_test = [int(float(x[1])) for x in test_edges];
d_test = [int(float(x[2])) for x in test_edges];
print '   %.2f seconds elapsed'%(time.clock()-st);

