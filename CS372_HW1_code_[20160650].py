import nltk
import time
from nltk.corpus import reuters
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
now = time.time()
# To analyze better we use wordnet Lemmatizer
# and the corpus we use is the Reuters corpus
wnl = nltk.WordNetLemmatizer()
reuter_text = nltk.Text(reuters.words())

def similarity_likelihood(w1, tuple1):
    # a test function to see if our triple satisfies the standards
    # the triple is packed as w1, (w2, w3), e.g. 'extol', ('praise', 'highly')
    w2, w3 = tuple1
    # use synsets to get the definition string 
    s1 = wn.synsets(w1)
    s2 = wn.synsets(w2)
    s3 = wn.synsets(w3)
   
    if wnl.lemmatize(w1) == wnl.lemmatize(w2) or wnl.lemmatize(w1) == wnl.lemmatize(w3):
        # we want similar phrases, not phrases with essentially same words.
        return False

    # get all possible part of speeches each word can have
    w1_pos = set([sset.pos() for sset in s1])
    w2_pos = set([sset.pos() for sset in s2])
    w3_pos = set([sset.pos() for sset in s3])
    
    sset_list = [sset1 for sset1 in s1]
    if set('n') == w1_pos:
        # we at least want to have a possibility of w1 not being a noun.
        # we do check this below again, but for early detection we add this step.
        return False

    excellence = False
    # for each synset in synsets...
    for sset in sset_list:
        # ...get the part of speech...
        target_pumsa = sset.pos()
        # ...and if the synset is not a noun...
        if not target_pumsa == 'n':
            #...get the definition string of the synset...
            defs = sset.definition()
            #...where if w2 or w3 is in the describing string and might have the same part of speech of w1, while the other one has a possibility of being an adverb
            if w2 in defs :
                if len(set(target_pumsa) & w2_pos) > 0 and 'r' in w3_pos:
                    excellence = True
            elif w3 in defs:
                if len(set(target_pumsa) & w3_pos) > 0 and 'r' in w2_pos:
                    excellence = True
    return excellence

stopword = stopwords.words()

print("precomputing... \nto inform you the progress, the numbers will count up to %1.1f million, twice." %(len(reuter_text) / (10 ** 6)))

# we want to find pairs of w1 and (w2, w3) so that there exists two words w_a and w_b such that both strings (w_a + w1 + w_b) and (w_a + w2 + w3 + w_b) exist in the corpus.
# so we make a python dictionary of 3-consecutive words and 4-consecutive words, where the key is the first and last word pair and the value of the key is the middle word(s).
# further we just discard non-alphabetic tokens and stopwords to improve quality.
trigrams = {}
for i in range(len(reuter_text) - 2):
    if (i % 100000 == 0):
        print(i)
    w1, w2, w3 = reuter_text[i: i+3]
    if w2.isalpha() and not w2.lower() in stopword:
        w1 = w1.lower()
        w2 = w2.lower()
        w3 = w3.lower()
        if (w1, w3) in trigrams:
            trigrams[(w1, w3)] = trigrams[(w1, w3)] | set([w2])
        else :
            trigrams[(w1, w3)] = set([w2])

quadgrams = {}
for i in range(len(reuter_text) - 3):
    if (i % 100000 == 0):
        print(i)
    w1, w2, w3, w4 = reuter_text[i: i+4]
    #print(w1, w2, w3)
    if w2.isalpha() and w3.isalpha() and (not w2.lower() in stopword) and (not w3.lower() in stopword) :
        w1 = w1.lower()
        w2 = w2.lower()
        w3 = w3.lower()
        w4 = w4.lower()
        if (w1, w4) in trigrams:
            if (w1, w4) in quadgrams:
                quadgrams[(w1, w4)] = quadgrams[(w1, w4)] | set([(w2, w3)])
            else :
                quadgrams[(w1, w4)] = set([(w2, w3)])

# from dictionaries made we find for a match; this and the previous step is necessarily finding w1, (w2, w3) pairs with the same context.
res_list = []
search_table = dict()
inverse_search_table = dict()
print("%d keys to test are found, please be patient." %( len(list(quadgrams.keys())) ))

# for those matching pairs with the same context we use the test function defined above to see if they are 'synomyms' in the sense of the test function result.
for key in list(quadgrams.keys()):
    tests = [(target, bullet) for target in trigrams[key] for bullet in quadgrams[key]]
    for test in tests:
        w1, tuple1 = test
        
        # to avoid superfluous overlapping, if w1 or (w2, w3) pair is already in the result list we reject this test case ...
        if w1 in search_table:
            break
        if tuple(sorted(tuple1)) in inverse_search_table:
            break
        w2, w3 = sorted(tuple1)
        
        # ...and also the case where w2 or w3 is not indisputably an adverb.
        # This step could be merged with the test function, but as an effort to reduce the running time of this code, checking such is done before calling the test function.
        if (not set([sset.pos() for sset in wn.synsets(w2)]) == set('r')) and (not set([sset.pos() for sset in wn.synsets(w3)]) == set('r')):
            break
        
        # finally the test function. 
        if similarity_likelihood(w1, tuple1):
            res_list.append(test)
            search_table[w1] = tuple1
            inverse_search_table[tuple(sorted(tuple1))] = w1

# print out first 50 results of triples
fout = open("./CS372_HW1_output_[20160650].csv", 'w')
res = res_list[:50]
for triple in res :
    w1, w23 = triple
    w2, w3  = w23
    print(w1 + ',' + w2 + ',' + w3, file = fout)
fout.close()


# among results we find those words which are "purely" adverbs, and print out the results according to their frequency of appearing in full result list.
describing = [w for tup in list(inverse_search_table)\
              for w in list(tup)\
              if set([sset.pos() for sset in wn.synsets(w)]) == set('r')]
fd = nltk.FreqDist(describing)
adverbs = [adv for adv, _ in list(fd.most_common())]
print("candidates of intensity-modifying verbs : ", adverbs[:min(len(adverbs), 50)])
print("elapsed : %.6f" %(time.time() - now))

