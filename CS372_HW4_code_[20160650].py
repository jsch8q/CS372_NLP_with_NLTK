import nltk, csv, re
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
wnl = nltk.WordNetLemmatizer()

target_verbs = ['activate', 'inhibit', 'bind', 'stimulate', 'abolish']

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
    '''
        removes explanatory parts (wrapped in round parantheses)
        from the sentence.
    '''
    while '(' in sent:
        a = sent.index('(')
        b = sent.index(')', a)
        sent = sent[:a] + sent[b+1:]
    sent = re.sub(r"\s+", " ", sent)
    return sent

def remove_hyphen(pos_sent, tokens):
    '''
        word tokenizers / POS taggers often considers
        hyphenated words in the form "A-B" as three-word clause,
        resulting in 'A', '-', and 'B'. This reverts this back.
    '''
    while '-' in pos_sent:
        i = pos_sent.index('-')
        pos_sent = pos_sent[:i-1] + pos_sent[i+1:]
        tokens[i+1] = ''.join(tokens[i-1:i+2])
        tokens = tokens[:i-1] + tokens[i+1:]
    return pos_sent, tokens

def remove_slash(pos_sent, tokens):
    '''
        analogous to remove_hyphen, but this function deals with
        slashed words, in which units like 'm/s' also falls into
        this category.
    '''
    while '/' in tokens:
        i =tokens.index('/')
        tokens[i+1] = ''.join(tokens[i-1:i+2])
        tokens = tokens[:i-1] + tokens[i+1:]
        pos_sent = pos_sent[:i-1] + pos_sent[i+1:]
    return pos_sent, tokens

def parse_list_in_str_format(liststr):
    '''
        helper function to read csv file.
        reads hand-annotated expected results.
    '''
    l = []
    while '(' in liststr:
        #print(liststr)
        a = liststr.index('(')
        b = liststr.index(')', a)
        l.append(tuple(liststr[a+1:b].split(',')))
        liststr = liststr[b:]
    return l

def emergency_find_triple(words, pos_sent):
    '''
        We hope that the grammer works well on the sentences,
        but life is not always so easy. For ill-POS-tagged
        sentences, such as the cases where "target verbs" are
        mistagged as nouns, the grammer will fail parsing. There
        are also weird sentences which does not fit in my grammer.
        For such cases, this ad-hoc emergency extraction is executed.
    '''
    idx = []
    res = []
    for i in range(len(words)):
        word = words[i]
        #find target verbs
        if word[-3:] != 'ing' and words[i-1].lower() != 'to'\
           and wnl.lemmatize(word, 'v') in target_verbs:
            idx.append(i)

    # heuristic set of POS tags to capture
    entity_pos = {'XX', 'SYM', 'ADD', 'CD', 'EX', 'FW', 'GW', '$', 'NIL', 'NN',\
                  'NNP', 'NNS', 'NNPS', 'PRP', 'PRP$', 'AFX', 'JJ', 'JJR', 'JJS',\
                  'POS', 'CC', 'DT', 'PDT'}
    for i in idx:
        a = i + 1
        while pos_sent[a] in entity_pos:
            a = a + 1
        Ylist = words[i+1 : a]
        b = i - 1
        while pos_sent[b] in entity_pos:
            b = b - 1
        Xlist = words[b+1 : i]
        
        # concatenate captured words and modify formats
        X = ' '.join(Xlist)
        Y = ' '.join(Ylist)
        X = re.sub(' - ', '-', X)
        Y = re.sub(' - ', '-', Y)
        X = re.sub(' / ', '/', X)
        Y = re.sub(' / ', '/', Y)
        
        res.append((X, words[i], Y))
    return res
                            
def extract_from_tree(words, pos_sent, tr):
    '''
        A tree is successfully made using the grammar.
        We use the parsed result, and extract triples following
        along the tree structure.
    '''
    idx = []
    res = []
    for i in range(len(words)):
        word = words[i]
        if word[-3:] != 'ing' and words[i-1].lower() != 'to'\
           and wnl.lemmatize(word, 'v') in target_verbs:
            idx.append(i)
    leaves_position = []
    q = [i for i in tr.treepositions()]
    for i in range(len(q)-1):
        if list(q[i]) != list(q[i+1])[:-1] :
            leaves_position.append(q[i])
    leaves_position.append(q[-1])

    for i in idx:
        # First, we determine which subtree we should look at.
        # To do so, we find where our target verbs are, and using that
        # position information, we take one leaf node which is highly
        # likely to be in the subtree containing X and Y in the triple
        # <X action Y>. 
        highest_V = leaves_position[i]
        highest_V = tuple(list(highest_V)[:-1])
        while len(highest_V) != 0 and tr[highest_V].label() != 'V':
            highest_V = tuple(list(highest_V)[:-1])
        if len(highest_V) == 0:
            highest_V = leaves_position[i]
            highest_V = tuple(list(highest_V)[:-1])
            while len(highest_V) != 0 and tr[highest_V].label()[0] != 'J':
                highest_V = tuple(list(highest_V)[:-1])
            if len(highest_V) == 0:
                highest_V = (0)
        if tuple(list(highest_V) + [1]) in q:
            Y_start = leaves_position[i+2]
            X_end   = leaves_position[i-2]
        else :
            Y_start = leaves_position[i+1]
            X_end   = leaves_position[i-1]

        # Next we get the path from the leaf nodes to the root
        X_to_top = []
        Y_to_top = []
        verb_to_top = []

        # To find the lowest common ancestor of (X, action),
        # and the lowest common ancestor of (action, Y)
        tmp = X_end
        while len(tmp) > 0:
            X_to_top.append(tuple(list(tmp)[:-1]))
            tmp = tuple(list(tmp)[:-1])
        tmp = Y_start
        while len(tmp) > 0:
            Y_to_top.append(tuple(list(tmp)[:-1]))
            tmp = tuple(list(tmp)[:-1])
        tmp = highest_V
        while len(tmp) > 0:
            verb_to_top.append(tuple(list(tmp)[:-1]))
            tmp = tuple(list(tmp)[:-1])
            
        X_verb_common_root_idx = -1
        try:
            while X_to_top[X_verb_common_root_idx] == verb_to_top[X_verb_common_root_idx]:
                X_verb_common_root_idx -= 1
        except(IndexError):
            X_verb_common_root_idx = -1
        X_verb_common_root_idx += 1
        X_verb_common_root = verb_to_top[X_verb_common_root_idx]

        Y_verb_common_root_idx = -1
        try:
            while Y_to_top[Y_verb_common_root_idx] == verb_to_top[Y_verb_common_root_idx]:
                Y_verb_common_root_idx -= 1
        except(IndexError):
            Y_verb_common_root_idx = -1
        Y_verb_common_root_idx += 1
        Y_verb_common_root = verb_to_top[Y_verb_common_root_idx]

        # According to grammar, and analyzing train data results, these labels are the
        # ones to find while going down the tree from the least common ancestor.
        target_label_set = {'Snopunc', 'VO', 'JP', 'NP', 'NPP', 'Det', 'Nand', \
                                           'Jand', 'NPPand', 'Andor', 'N', 'J', 'Adj', 'CC', 'DT'}

        # from the common ancestor of Y and action, we extract all leaf nodes
        # which will be conatined in Y by recursively looking at the subtrees
        # with the label in the target_label_set
        queue = [Y_verb_common_root]
        do_again = True
        while do_again:
            new_queue = []
            for pos in queue:
                if not tuple(list(pos) + [0]) in q:
                    new_queue.append(pos)
                else:
                    if tr[pos].label() in target_label_set:
                        new_queue.append(tuple(list(pos) + [0]))
                        if tuple(list(pos) + [1]) in q:
                            new_queue.append(tuple(list(pos) + [1]))
            if queue == new_queue:
                do_again = False
            else :
                queue = new_queue

        # and join the words in the leaf nodes to get Y
        yidx = [leaves_position.index(que) for que in queue if leaves_position.index(que) > i]
        y_words = [words[yx] for yx in yidx]
        Y = ' '.join(y_words)

        # do the same thing with X
        queue = [X_verb_common_root]
        do_again = True
        while do_again:
            new_queue = []
            for pos in queue:
                if not tuple(list(pos) + [0]) in q:
                    new_queue.append(pos)
                else:
                    if tr[pos].label() in target_label_set:
                        new_queue.append(tuple(list(pos) + [0]))
                        if tuple(list(pos) + [1]) in q:
                            new_queue.append(tuple(list(pos) + [1]))
            if queue == new_queue:
                do_again = False
            else :
                queue = new_queue

        xidx = [leaves_position.index(que) for que in queue if leaves_position.index(que) < i]
        x_words = [words[xx] for xx in xidx]
        X = ' '.join(x_words)

        # re-format the results
        X = re.sub(' - ', '-', X)
        Y = re.sub(' - ', '-', Y)
        X = re.sub(' / ', '/', X)
        Y = re.sub(' / ', '/', Y)
        res_tup = (X, words[i], Y)
        res.append(res_tup)

    return res


full_data = []
sents = []
annots = []


# read input data
with open("./CS372_HW4_output_[20160650].csv", newline = '', encoding = 'utf-8') as csvfile:
    fin = csv.reader(csvfile, dialect='excel')
    for row in fin:
        full_data.append(row)
        sents.append(normalize_sents(row[0]))
        annots.append(parse_list_in_str_format(row[-1]))

# grammar built from 80 train sentences
MEDgrammar = nltk.CFG.fromstring("""
S 	-> Snopunc Punc
Snopunc -> NP VP | NP VO | NPP VP | NPP VO | RP Snopunc | Snopunc RP | Snopunc Ssub | Det VO | Det VP
Ssub 	-> Punc Snopunc
Sbar 	-> Andor Snopunc | Rwh Snopunc | P Snopunc
RP 	-> RP RP | RP Punc | Punc RP | Punc JP | Rand RP | Rwh VO | Rwh VP | 'HYPH' RP | PP Punc | R | Sbar
JP 	-> RP JP | JP RP | Adj | Jand JP | JP PP | NP POS | 'HYPH' JP | NP 'VBD'
NP 	-> JP NP | 'VBN' NP | Det NP | NP NP | Nand NP | 'HYPH' NP | NP JP | N
NPP 	-> NP PP | JP NPP | Det NPP | NPPand NPP
Vpp     -> Vpp Vpp | 'VBN'  
Vpass   -> V Vpp
VP 	-> V | RP VP | Vand VP | 'HYPH' VP
VO 	-> VP NP | VP JP | VP Sbar | VP NPP | VP PP | RP VO | VOand VO
PP 	-> P NP | PP PP | P VP | P VO | P 'VBN' | PPand PP | P Snopunc
Nand 	-> NP Punc | Nand Punc | NP Andor
Vand 	-> VP Punc | Vand Punc | VP Andor
Jand 	-> JP Punc | Jand Punc | JP Andor
Rand 	-> RP Punc | Rand Punc | RP Andor
PPand 	-> PPand Punc | PP Andor
VOand 	-> VOand Punc | VO Andor
NPPand 	-> NPPand Punc | NPP Andor
Det 	-> 'DT' | 'PDT'
N       -> 'XX' | 'SYM' | 'ADD' | 'CD' | 'EX' | 'FW' | 'GW' | '$' | 'NIL' | 'NN' | 'NNP' | 'NNS' | 'NNPS' | 'PRP' | 'PRP$' | 'VBN'
Adj     -> 'AFX' | 'JJ' | 'JJR' | 'JJS' | TO 'VB' | 'VBG' | 'VBN'
V       -> Vpass 'IN' | 'VBD' | 'VBP' | 'VBZ' | 'VB'
P 	-> 'IN' | 'RP' 
R       -> 'LS' | 'MD' | 'RB' | 'RBR' | 'RBS' | 'UH'
Rwh     -> 'WDT' | 'WP' | 'WP$' | 'WRB'
TO      -> 'TO'
POS     -> 'POS'
Andor   -> 'CC'
Punc 	-> ',' | '.' | "``" | "''" | '-LRB-' | '-RRB-' | ":" | 'NFP' 
#####################
#####################
#N 	-> 'degradation' | 'jelly' | 'proteases' | 'ADAMTS' | 'Tie2' | 'expression' | 'Angpt1' | 'we' | 'QS' | 'program' | 'multicellularity' | 'log' | 'Vibrio' | 'cholerae' | 'ATF4' | 'transcription' | 'RNA' | 'polymerase' | 'II' | 'region' | 'heterodimer' | 'subunits' | 'involving' | 'Mediator' | 'which' | 'activation' | 'genes'
#Adj 	-> 'proper' | 'cardiac' | 'crucial' | 'endocardial' | 'myocardial' | 'new' | 'frightened' | 'little' | 'tall' | 'α-like' | 'fast' | 'effective' | 'desired'
#V 	-> 'show' | 'regulates' | 'said' | 'thought' | 'put' | 'activates' | 'contacting' | 'provides'
#Vspecial -> 'activates'
#RB 	-> 'positively' | 'Here' | 'directly' | 'highly' | 'Mechanistically'
""") 

TP = 0
TF = 0
FP = 0
FN = 0


parser = nltk.ChartParser(MEDgrammar)
for i in range(80):
    print("sentence %d" %(i))
    pos_sent = [pos for (w, pos) in pos_tag(sents[i])]
    words = [w for (w, pos) in pos_tag(sents[i])]
    pos_sent, words = remove_hyphen(pos_sent, words)
    pos_sent, words = remove_slash(pos_sent, words)
    tree_list = [t for t in parser.parse(pos_sent)]

    # check : do we have problem in POS tagging?
    pos_tag_check_list = []
    for j in range(len(words)):
        word = words[j]
        if word[-3:] != 'ing' and words[j-1].lower() != 'to'\
           and wnl.lemmatize(word, 'v') in target_verbs:
            pos_tag_check_list.append(pos_sent[j])
    emergency = False
    for tag in pos_tag_check_list:
        if tag[0] != 'V':
            # Yes, POS tagging has a problem.
            emergency = True
            
    if emergency :
        # This case, the POS tagger did something wrong.
        extracted_res = set(emergency_find_triple(words, pos_sent))
    elif len(tree_list) == 0:
        # This case, the grammar failed to build the tree.
        extracted_res = set(emergency_find_triple(words, pos_sent))
    else :
        # Had no problem until now.
        extracted_res = set(extract_from_tree(words, pos_sent, tree_list[0]))

    try : 
        expect = annots[i]
        expected = []
        for (x,act,y) in expect:
            expected.append((x.strip(), act.strip(), y.strip()))
        expected_res = set(expected)
    except:
        expected_res = set(annots[i])
    print("expected : ", expected_res )
    print("extracted : ", extracted_res )

    TP = TP + len(expected_res.intersection(extracted_res))
    FN = FN + len(expected_res.difference(extracted_res))
    FP = FP + len(extracted_res.difference(expected_res))

prec = TP / (TP + FP)
reca = TP / (TP + FN)
print('precision : ', prec)
print('recall : ' , reca)
print('F-score : ', (2 * prec * reca) / (prec + reca)) 

TP = 0
TF = 0
FP = 0
FN = 0

for i in range(80, 100):
    print("sentence %d" %(i))
    pos_sent = [pos for (w, pos) in pos_tag(sents[i])]
    words = [w for (w, pos) in pos_tag(sents[i])]
    pos_sent, words = remove_hyphen(pos_sent, words)
    pos_sent, words = remove_slash(pos_sent, words)
    tree_list = [t for t in parser.parse(pos_sent)]

    # check : do we have problem in POS tagging?
    pos_tag_check_list = []
    for j in range(len(words)):
        word = words[j]
        if word[-3:] != 'ing' and words[j-1].lower() != 'to'\
           and wnl.lemmatize(word, 'v') in target_verbs:
            pos_tag_check_list.append(pos_sent[j])
    emergency = False
    for tag in pos_tag_check_list:
        if tag[0] != 'V':
            # Yes, POS tagging has a problem.
            emergency = True
            
    if emergency :
        # This case, the POS tagger did something wrong.
        extracted_res = set(emergency_find_triple(words, pos_sent))
    elif len(tree_list) == 0:
        # This case, the grammar failed to build the tree.
        extracted_res = set(emergency_find_triple(words, pos_sent))
    else :
        # Had no problem until now.
        extracted_res = set(extract_from_tree(words, pos_sent, tree_list[0]))

    try : 
        expect = annots[i]
        expected = []
        for (x,act,y) in expect:
            expected.append((x.strip(), act.strip(), y.strip()))
        expected_res = set(expected)
    except:
        expected_res = set(annots[i])
    print("expected : ", expected_res )
    print("extracted : ", extracted_res )

    TP = TP + len(expected_res.intersection(extracted_res))
    FN = FN + len(expected_res.difference(extracted_res))
    FP = FP + len(extracted_res.difference(expected_res))

prec = TP / (TP + FP)
reca = TP / (TP + FN)
print('precision : ', prec)
print('recall : ' , reca)
print('F-score : ', (2 * prec * reca) / (prec + reca)) 
