import nltk
from collections import defaultdict
wnl = nltk.WordNetLemmatizer()

#two different lemmatizing helper functions according to input pos
def lemmatize_tuple_verb(tup):
    li = []
    for w in tup:
        li.append(wnl.lemmatize(w,'v'))
    return tuple(li)

def lemmatize_tuple_noun(tup):
    li = []
    for w in tup:
        li.append(wnl.lemmatize(w,'n'))
    return tuple(li)

#checks if token is numerically modifying, i.e. 3rd-degree, 4-feet, ...
def is_numerical_mod(w):
    return True if w[0].isdigit() else False

#our corpus is pos-tagged brown corpus, and 'words' contain normalized tokens
tgwrd = nltk.corpus.brown.tagged_words(tagset = 'universal')
words = [w.lower() for w in nltk.corpus.brown.words()]

#pairs of the form ...(adverb) (verb or adj)... in the corpus 
adv_targ = [(tgwrd[i][0].lower(), tgwrd[i+1][0].lower()) \
              for i in range(len(tgwrd) - 1) \
              if ( tgwrd[i][1] == "ADV" \
                   and (tgwrd[i+1][1] == "VERB" or tgwrd[i+1][1] == "ADJ")\
                   and not (tgwrd[i-1][1] == "VERB" or tgwrd[i-1][1] == "ADJ") 
                   )]

#pairs of the form ...(verb or adj) (adverb)... in the corpus 
targ_adv = [(tgwrd[i-1][0].lower(), tgwrd[i][0].lower()) \
              for i in range(1, len(tgwrd)) \
              if ( tgwrd[i][1] == "ADV" \
                   and not (tgwrd[i+1][1] == "VERB" or tgwrd[i+1][1] == "ADJ")\
                   and (tgwrd[i-1][1] == "VERB" or tgwrd[i-1][1] == "ADJ") 
                   )]

#similar lists for adjectives and nouns
adj_targ = []
targ_adj = []
for i in range(1, len(tgwrd) - 1):
    if ( tgwrd[i][1] == "ADJ" and tgwrd[i+1][1] == "NOUN" ):
        # case where ...(adjective) (noun)...
        adj_targ.append( (tgwrd[i][0].lower(), tgwrd[i+1][0].lower()) )
    else :
        #for other cases as predicative use of adjectives,
        #finding for the closest noun in front of the adjective 
        j = i - 1
        w = tgwrd[j]
        while (j > 0) and not (w[1] in {'NOUN', 'X', '.'}):
            j -= 1
            w = tgwrd[j]
        if j == 0 or w[1] != 'NOUN':
            # exception catch fail, skip this case :
            # punctuation or token of unknown pos is encountered before any nouns
            pass 
        else :
            # exception catch success : add to list
            targ_adj.append( (tgwrd[j][0].lower(), tgwrd[i][0].lower()) )
    

freqdict = defaultdict(set)
pairdict = defaultdict(set)
# add lemmatized found pairs to defaultdict to analyze frequencies
# boolean values indicate the order of the modifying and modified words
# but in the case of predicative adjectives, when showing the pair only
# it is natural to appear in the order of (adjective, noun) so we force such ordering.
for tup in targ_adv:
    (targ, adv) = lemmatize_tuple_verb(tup)
    pairdict[adv].add((True, targ))
    freqdict[adv].add(targ)
for tup in adv_targ:
    (adv, targ) = lemmatize_tuple_verb(tup)
    pairdict[adv].add((False, targ))
    freqdict[adv].add(targ)
for tup in targ_adj:
    (targ, adj) = lemmatize_tuple_noun(tup)
    pairdict[adj].add((False, targ))
    freqdict[adj].add(targ)
for tup in adj_targ:
    (adj, targ) = lemmatize_tuple_noun(tup)
    pairdict[adj].add((False, targ))
    freqdict[adj].add(targ)

freqlist = []
for adv in list(freqdict):
    if len(pairdict[adv]) > 1 and adv.isalpha():
        # score (sort criteria):
        #   1st priority : the number of the words modified by the adj or adv
        #   2nd priority : number of distinct pairs containing the adj or adv
        #   3rd priority : alphabetical order
        #   exception : if adj or adv occurs only once,
        #               then to avoid the possibility of the word being a compound
        #               word or a just a rarely used word, we drop those cases.
        freqlist.append((len(freqdict[adv]), adv))

descrlist = [w[1] for w in sorted(freqlist)]

result = []

#recovering pairs and storing all in a single list
for descr in descrlist:
    targs = list(freqdict[descr])
    for targ in targs :
        lemtarg = wnl.lemmatize(targ)
        if (False, lemtarg) in pairdict[descr]:
            tup = (descr, targ)
        else :
            tup = (targ, descr)
        result.append(tup)

fin = open('./CS372_HW2_output_[20160650].csv', 'w')
for pair in result[:100]:
    a, b = pair
    print(a + ',' + b, file = fin) 
#print(result[:100])
fin.close()
#lemmatize entries...
#add adjectives... assuming simple rule on predicative adjectives
#maybe we should remove once appearings... they may be just rare words. # Imshiro done
