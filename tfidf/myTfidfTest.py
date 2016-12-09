import math
import tfidf
import json
import time
import sys, os
sys.path.append('/home/hltcoe/gsell/tools/python_mods/');
review_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_review.json';
DEFAULT_IDF_UNITTEST = 1.0
test_cond = '12_45'
start = time.clock();

execfile('../load_edge_attr_data.py');
#to_save['Train Reviewer List'] = train_reviewer_lst;
#to_save['Test Reviewer List'] = test_reviewer_lst;
#to_save['Business Information'] = bid_info;
#to_save['Review Information'] = review_info;
#to_save['Reviewer Reviews'] = reviewer_reviews;
#to_save['Reviewed Business List'] = reviewed_bid_lst;
B = len(data['Reviewed Business List'])
R = len(data['Train Reviewer List'])
I = len(data['Review Information'])
print '   %.2f seconds elapsed'%(time.clock()-start);

print '\nBuilding index lookups...'
reviewer_idx = {}
business_idx = {}
for n in range(R) : reviewer_idx[data['Train Reviewer List'][n]] = n
for n in range(B) : business_idx[data['Reviewed Business List'][n]] = n
print '   %.2f seconds elapsed'%(time.clock()-start);

print '\nBuilding business review idx'
business_revs = {};
for uid in reviewer_idx :
    for rid in data['Reviewer Reviews'][uid] :
        reviewInfo = data['Review Information'][rid];
        bid = reviewInfo['business_id'];

        if bid not in business_revs :
            business_revs[bid] = {};
        if rid not in business_revs[bid] :
            business_revs[bid][rid] = reviewInfo['text'];
print '    %.2f seconds elapsed'%(time.clock()-start);


# Try parsing the review 'text' into the tfidf
total_tfidf = tfidf.TfIdf("tfidf_teststopwords.txt")
numReviews = len(review_info)
counter = 0
for bid in business_revs:
    # Create a new TF-IDF instance for each business reviewed
    counter += 1
    business_tfidf = tfidf.TfIdf("tfidf_teststopwords.txt")
    business_doc = ''
    for rid in business_revs[bid] :
        reviewinfo = business_revs[bid][rid]
        business_tfidf.add_input_str(reviewinfo)
        business_doc += reviewinfo

    total_tfidf.add_input_str(business_doc)
    # Give relevant info for the TF-IDF corpus used and give the stats 
    print '\nBusiness: ' + str(bid) + ' (' + str(counter) + '/' + str(len(business_revs)) + ')\tNum Docs: ' + str(business_tfidf.get_num_docs()) + '/' + str(numReviews) + '\tNum Words: ' + str(len(set(business_tfidf.return_tokens())))

tokens = total_tfidf.return_tokens()
keywords = total_tfidf.get_doc_keywords()
for word in keywords[:10] : 
    print "\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens[word[0]],business_tfidf.get_idf(word[0]),word[1])
print '   %.2f minutes elapsed -- (%.2f sec)'%((time.clock()-start)/60, time.clock()-start);

