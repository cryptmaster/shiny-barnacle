import math
import tfidf
import json
import time
import sys, os
sys.path.append('/home/hltcoe/gsell/tools/python_mods/');
review_file = '/home/hltcoe/vlyzinski/yelp/yelp_academic_dataset_review.json';
DEFAULT_IDF_UNITTEST = 1.0
test_cond = '1_5'
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


# Test IFIDF on files from terminal
#my_tfidf = tfidf.TfIdf(textFile, "tfidf_teststopwords.txt", DEFAULT_IDF = DEFAULT_IDF_UNITTEST)
#tokens = my_tfidf.return_tokens()
#tokens_set = set(tokens)
#keywords = my_tfidf.get_doc_keywords()
#print 'Num Docs: ' + str(my_tfidf.get_num_docs())
#print 'Num Words: ' + str(len(tokens_set))
#for word in keywords : 
#    print "\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens[word[0]],my_tfidf.get_idf(word[0]),word[1])
#print '   %.2f seconds elapsed'%(time.clock()-start);



# Try parsing the review 'text' into the tfidf
numReviews = len(review_info)
for bid in business_revs:
    # Create a new TF-IDF instance for each business reviewed
    my_tfidf = tfidf.TfIdf("tfidf_teststopwords.txt")
    for rid in business_revs[bid] :
        reviewinfo = business_revs[bid][rid]
        my_tfidf.add_input_str(reviewinfo)

    # Give relevant info for the TF-IDF corpus used and give the stats
    print '\tNum Docs: ' + str(my_tfidf.get_num_docs()) + '/' + str(numReviews) + '\tNum Words: ' + str(len(set(my_tfidf.return_tokens())))
    tokens = my_tfidf.return_tokens()
    keywords = my_tfidf.get_doc_keywords()
    for word in keywords[:10] : 
        print "\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens[word[0]],my_tfidf.get_idf(word[0]),word[1])
print '   %.2f seconds elapsed'%(time.clock()-start);

