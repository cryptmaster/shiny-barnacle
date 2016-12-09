#!/usr/bin/env python
# 
# Copyright (C) 2009.  All rights reserved.

__author__ = "Niniane Wang"
__email__ = "niniane at gmail dot com"

import math
import tfidf
import unittest

DEFAULT_IDF_UNITTEST = 1.0

def get_expected_idf(num_docs_total, num_docs_term):
    print 'Predic IDF --- Num Docs total: ' + str(num_docs_total) + '\tNum Docs Term: ' + str(num_docs_term) + '\tIDF: ' + str(math.log(float(1 + num_docs_total) / (1 + num_docs_term))) 
    return math.log(float(1 + num_docs_total) / (1 + num_docs_term))

class TfIdfTest(unittest.TestCase):
    def testGetIdf(self):
        print "\n====== Test querying the IDF for existent and nonexistent terms. ======"
        my_tfidf = tfidf.TfIdf("tfidf_testcorpus.txt", DEFAULT_IDF = DEFAULT_IDF_UNITTEST)

        # Test querying for a nonexistent term.
        self.assertEqual(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("nonexistent"))
        self.assertEqual(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("THE"))

        self.assertTrue(my_tfidf.get_idf("a") > my_tfidf.get_idf("the"))
        self.assertAlmostEquals(my_tfidf.get_idf("girl"), my_tfidf.get_idf("moon"))

    def testKeywords(self):
        print "\n====== Test retrieving keywords from a document, ordered by tf-idf. ======"
        my_tfidf = tfidf.TfIdf("tfidf_testcorpus.txt", DEFAULT_IDF = 0.01)

        # Test retrieving keywords when there is only one keyword.
        keywords = my_tfidf.get_doc_keywords("the spoon and the fork")
        self.assertEqual("the", keywords[0][0])

        # Test retrieving multiple keywords.
        keywords = my_tfidf.get_doc_keywords("the girl said hello over the phone")
        self.assertEqual("girl", keywords[0][0])
        self.assertEqual("phone", keywords[1][0])
        self.assertEqual("said", keywords[2][0])
        self.assertEqual("the", keywords[3][0])

    def testAddCorpus(self):
        print "\n====== Test adding input documents to the corpus. ======"
        my_tfidf = tfidf.TfIdf("tfidf_testcorpus.txt", DEFAULT_IDF = DEFAULT_IDF_UNITTEST)
  
        self.assertEquals(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("water"))
        #self.assertAlmostEquals(get_expected_idf(my_tfidf.get_num_docs(), 1), my_tfidf.get_idf("moon"))
        get_expected_idf(my_tfidf.get_num_docs(), 1)
        my_tfidf.get_idf("moon")
        self.assertAlmostEquals( get_expected_idf(my_tfidf.get_num_docs(), 5), my_tfidf.get_idf("said"))

        my_tfidf.add_input_document("water, moon")
 
        self.assertAlmostEquals(get_expected_idf(my_tfidf.get_num_docs(), 1), my_tfidf.get_idf("water"))
        self.assertAlmostEquals(get_expected_idf(my_tfidf.get_num_docs(), 2), my_tfidf.get_idf("moon"))
        self.assertAlmostEquals(get_expected_idf(my_tfidf.get_num_docs(), 5), my_tfidf.get_idf("said"))

    def testNoCorpusFiles(self):
        print "\n====== Test with no input documents in the corpus. ======"
        my_tfidf = tfidf.TfIdf(DEFAULT_IDF = DEFAULT_IDF_UNITTEST)

        self.assertEquals(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("moon"))
        self.assertEquals(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("water"))
        self.assertEquals(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("said"))

        my_tfidf.add_input_document("moon")
        my_tfidf.add_input_document("moon said hello")

        self.assertEquals(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("water"))
        self.assertAlmostEquals(get_expected_idf(my_tfidf.get_num_docs(), 1), my_tfidf.get_idf("said"))
        self.assertAlmostEquals(get_expected_idf(my_tfidf.get_num_docs(), 2), my_tfidf.get_idf("moon"))

    def testStopwordFile(self):
        my_tfidf = tfidf.TfIdf("tfidf_testcorpus.txt", "tfidf_teststopwords.txt", DEFAULT_IDF = DEFAULT_IDF_UNITTEST)

        self.assertEquals(DEFAULT_IDF_UNITTEST, my_tfidf.get_idf("water"))
        self.assertEquals(0, my_tfidf.get_idf("moon"))
        self.assertAlmostEquals(get_expected_idf(my_tfidf.get_num_docs(), 5), my_tfidf.get_idf("said"))

        my_tfidf.add_input_document("moon")
        my_tfidf.add_input_document("moon and water")

        self.assertAlmostEquals(get_expected_idf(my_tfidf.get_num_docs(), 1), my_tfidf.get_idf("water"))
        self.assertEquals(0, my_tfidf.get_idf("moon"))
        self.assertAlmostEquals(get_expected_idf(my_tfidf.get_num_docs(), 5), my_tfidf.get_idf("said"))


def main():
#    suite = unittest.TestLoader().loadTestsFromTestCase(TfIdfTest)
#    unittest.TextTestRunner(verbosity=2).run(suite)

    my_tfidf = tfidf.TfIdf("tfidf_testcorpus.txt", DEFAULT_IDF = DEFAULT_IDF_UNITTEST)
#    print "for word \'moon\'"
#    my_tfidf.get_idf("moon")
#    my_tfidf.add_input_document("water, moon")
#    print "added another instance of moon"
#    my_tfidf.get_idf("moon")

    tokens = my_tfidf.get_tokens()
    print tokens
    keywords = my_tfidf.get_doc_keywords()
    tokens_set = set(tokens)
    print 'Words: ' + str(tokens)
    print 'Num Docs: ' + str(my_tfidf.get_num_docs())
    for word in keywords : 
        print "\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens.count(word[0]),my_tfidf.get_idf(word[0]),word[1])
 
    keywords = my_tfidf.get_str_keywords("the girl said hello over the phone")
    tokens = my_tfidf.get_tokens_str("the girl said hello over the phone")
    tokens_set = set(tokens)
    print 'Words: ' + str(tokens)
    print 'Num Docs: ' + str(my_tfidf.get_num_docs())
    for word in keywords : 
        print "\tWORD: %s\tTF:%.0f\tIDF:%.3f\tTF-IDF:%.3f" %(str(word[0]),tokens.count(word[0]),my_tfidf.get_idf(word[0]),word[1])
    


if __name__ == '__main__':
    main()
