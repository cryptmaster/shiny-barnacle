import math
import tfidf
import unittest
import sys, os
sys.path.append('/home/hltcoe/gsell/tools/python_mods/');

DEFAULT_IDF_UNITTEST = 1.0
textFile = sys.argv[1]
start = time.clock();

my_tfidf = tfidf.TfIdf(textFile, "tfidf_teststopwords.txt", DEFAULT_IDF = DEFAULT_IDF_UNITTEST)
#    my_tfidf.get_idf("moon")
#    my_tfidf.add_input_document("water, moon")
#    my_tfidf.get_idf("moon")

tokens = my_tfidf.return_tokens()
tokens_set = set(tokens)
keywords = my_tfidf.get_doc_keywords()

print 'Num Docs: ' + str(my_tfidf.get_num_docs())
print 'Num Words: ' + str(len(tokens_set))
for word in keywords : 
    print "\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens[word[0]],my_tfidf.get_idf(word[0]),word[1])
print '   %.2f seconds elapsed'%(time.clock()-start);

my_tfidf.add_input_document("textSample1.txt")

tokens = my_tfidf.return_tokens()
tokens_set = set(tokens)
keywords = my_tfidf.get_doc_keywords()

print 'Num Docs: ' + str(my_tfidf.get_num_docs())
print 'Num Words: ' + str(len(tokens_set))
for word in keywords[:10] : 
    print "\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens[word[0]],my_tfidf.get_idf(word[0]),word[1])
print '   %.2f seconds elapsed'%(time.clock()-start);


#keywords = my_tfidf.get_str_keywords("the girl said hello over the phone")
#tokens = my_tfidf.get_tokens_str("the girl said hello over the phone")
#tokens_set = set(tokens)
#print 'Words: ' + str(tokens)
#print 'Num Docs: ' + str(my_tfidf.get_num_docs())
#for word in keywords : 
#    print "\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens.count(word[0]),my_tfidf.get_idf(word[0]),word[1])
