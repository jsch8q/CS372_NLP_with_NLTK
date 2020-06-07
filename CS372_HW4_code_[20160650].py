import nltk, csv, re
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize

##################################
##### POS tagging with spaCy #####
##################################
'''
    We use spaCy ONLY to get better POS tagging results.
    I find it very confusing to install and use allennlp,
    which was the module suggested for POS tagging in the
    TA office hour. SpaCy is the alternative I've found.
'''

import spacy
nlp = spacy.load("en_core_web_md")
def pos_tag(sent):
    doc = nlp(sent)
    res = [(token.text, token.tag_) for token in doc]
    return res

##################################
##### POS tagging with NLTK ######
##################################

"""
from nltk import pos_tag as pos_tag_nltk
def pos_tag(sent):
    return pos_tag_nltk(word_tokenize(sent))
"""

##################################
#######  helper functions ########
##################################

def normalize_sents(sent):
    sent = re.sub(r"\(.*\)", "", sent)
    return sent

def parse_list_in_str_format(liststr):
    l = []
    while '(' in liststr:
        print(liststr)
        a = liststr.index('(')
        b = liststr.index(')', a)
        l.append(tuple(liststr[a:b].split(',')))
        liststr = liststr[b:]
    return l
    

full_data = []
sents = []
annots = []

with open("./MEDLINE.csv", newline = '', encoding = 'utf-8') as csvfile:
    fin = csv.reader(csvfile, dialect='excel')
    for row in fin:
        full_data.append(row)
        sents.append(normalize_sents(row[0]))
        annots.append(parse_list_in_str_format(row[-1]))

pos_tag_set = set()
for i in range(80):
    tmps = set([pos for (token, pos) in pos_tag(sents[i])])
    if 'XX' in tmps:
        print(pos_tag(sents[i]))
    pos_tag_set = pos_tag_set.union(tmps)

MEDgrammar = nltk.CFG.fromstring("""
S 	-> Snopunc Punc
Snopunc -> NP VP | NP VO | RP Snopunc | Snopunc RP | Snopunc Ssub
Ssub    -> Punc Snopunc
Sbar    -> IN Snopunc | Andor Snopunc
RP 	-> RP RP | RP Punc | Rand RP | RB
JP      -> RP JP | Adj | Jand JP | JP PP
NP 	-> JP NP | Det NP | NP NP | Nand NP | N
NPP     -> NP PP | JP NPP | Det NPP | NPPand NPP 
VP 	-> V | Vbe | RP VP | Vand VP
VO      -> VP NP | VP JP | VP Sbar | VP NPP | Vspecial NPP | RP VO | VOand VO
PP 	-> P NP | PP PP | P VP | P VO | PPand PP
Nand    -> NP Punc | Nand Punc | NP Andor
Vand    -> VP Punc | Vand Punc | VP Andor
Jand    -> JP Punc | Jand Punc | JP Andor
Rand    -> RP Punc | Rand Punc | RP Andor
PPand   -> PPand Punc | PP Andor
VOand   -> VOand Punc | VO Andor
NPPand  -> NPPand Punc | NPP Andor
Det 	-> 'the' | 'a'
N 	-> 'degradation' | 'jelly' | 'proteases' | 'ADAMTS' | 'Tie2' | 'expression' | 'Angpt1' | 'we' | 'QS' | 'program' | 'multicellularity' | 'log' | 'Vibrio' | 'cholerae' | 'ATF4' | 'transcription' | 'RNA' | 'polymerase' | 'II' | 'region' | 'heterodimer' | 'subunits' | 'involving' | 'Mediator' | 'which' | 'activation' | 'genes'
Adj 	-> 'proper' | 'cardiac' | 'crucial' | 'endocardial' | 'myocardial' | 'new' | 'frightened' | 'little' | 'tall' | 'α-like' | 'fast' | 'effective' | 'desired'
V 	-> 'show' | 'regulates' | 'said' | 'thought' | 'put' | 'activates' | 'contacting' | 'provides'
#Vspecial -> 'activates'
Vbe     -> 'is' | 'was' 
P 	-> 'on' | 'of' | 'in' | 'without' | 'for' | Pby
Pby -> 'by'  
RB 	-> 'positively' | 'Here' | 'directly' | 'highly' | 'Mechanistically'
IN 	-> 'that'
Andor   -> 'and' | 'or'
Punc 	-> ',' | '.'
""") 
# 1. 수동태 다룰 수 있게 해 놓으세요...
# 2. grammar 망했을 때 땜질용 extracter 필요해요
# 3. token 직접 넣는 것 보다 그냥 pos tag를 token처럼 다루는 게 편하지 않아요?

parser = nltk.ChartParser(MEDgrammar)
only_one = True
for tree in parser.parse(word_tokenize(sents[2])):
    if only_one:
        print(tree)
        only_one = False
    else :
        break
