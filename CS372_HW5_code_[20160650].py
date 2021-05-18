import nltk, re, pickle
import gap_scorer
from nltk.corpus import conll2000
cp = nltk.RegexpParser
from nltk import word_tokenize, sent_tokenize

print("It took about 3 minutes on my laptop. \nPlease be patient.")

##################################
##### POS tagging with spaCy #####
##################################
'''
    We use spaCy ONLY to get better POS tagging results.
'''

import en_core_web_md
nlp = en_core_web_md.load()

def pos_tag(sent):
    doc = nlp(sent)
    res = [(token.text, token.tag_) for token in doc]
    return res

##################################
##### POS tagging with NLTK ######
##################################
'''
    In the case where spaCy shall not be used,
    POS tagger provided by NLTK should be in use,
    by uncommenting the block below and commenting the block above.
'''
"""
from nltk import pos_tag as pos_tag_nltk
def pos_tag(sent):
    return pos_tag_nltk(word_tokenize(sent))
"""

#######################################
##### Building Chunker with NLTK ######
##### Following the lecture notes #####
#######################################

class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)

        self.classifier = nltk.MaxentClassifier.train(
            train_set, algorithm = 'gis', trace = 0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)

        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c)
                         for (w, t, c) in nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)
                        
def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else :
        prevword, prevpos = sentence[i-1]

    if i == len(sentence) - 1:
        nextword, nextpos = "<START>", "<START>"
    else :
        nextword, nextpos = sentence[i+1]
        
    return {"pos" : pos, "word" : word, "prevpos" : prevpos, "nextpos" : nextpos,
            "prev+pos" : "%s+%s" %(prevpos, pos), "pos+next" : "%s+%s" %(pos, nextpos),
            "tags-since-dt": tags_since_dt(sentence, i)}
                        
def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT' :
            tags = set()
        else :
            tags.add(pos)
    return '+'.join(sorted(tags))

##################################
##### Other helper functions #####
##################################

def get_position_indices(sents_list, parsed_forest, find_idx):
    '''
        given the offset with the text,
        and the list of chunking-parsed trees, i.e. the parsed forest,
        returns in which tree is the word with the offset, and the position information
        in the nltk.tree.Tree data structre.
    '''
    current = 0
    next_sent_start = 0
    sent_idx = 0
    for sent in sents_list:
        if next_sent_start > find_idx:
            break
        else :
            current = next_sent_start
            next_sent_start += (len(sent) + 1)
            sent_idx += 1

    sent_idx -= 1
    sent = sents_list[sent_idx]
    parsed_tree = parsed_forest[sent_idx]
    next_word_start = current

    for i in range(len(parsed_tree.leaves())):
        leafpos = parsed_tree.leaf_treeposition(i)
        token = parsed_tree[leafpos]
        if next_word_start > find_idx:
            break
        else:
            word, pos = token
            current = next_word_start
            next_word_start += len(word)
            sent = sent[len(word) : ]
            if sent[0] == ' ':
                sent = sent[1:]
                next_word_start += 1

    return (sent_idx, parsed_tree.leaf_treeposition(i-1))         

def phrases_position_list(parsed_tree):
    '''
        from nltk.tree.Tree, returns the position list of the children of the root node.
    '''
    return [tup for tup in parsed_tree.treepositions() if len(tup) == 1]

def constituent_list(parsed_tree, phrases_position_list):
    '''
        retruns the list of labels of the nodes which are children of the root node.
        That is, this function extracts a list containing NP, VP, PP, or the POS of the unchunked
        word token from the chunked sentence.
    '''
    l = []
    for tup in phrases_position_list:
        if type(parsed_tree[tup]) == nltk.tree.Tree:
            l.append(parsed_tree[tup].label())
        else :
            l.append(parsed_tree[tup][1])
    return l


def constituent_info(position_in_tree, parsed_tree):
    '''
        'S'ubject or 'O'bject? Complements are also classified as objects.
    '''
    ppl = phrases_position_list(parsed_tree)
    cl = constituent_list(parsed_tree, ppl)
    head_position = tuple([position_in_tree[0]])
    head_idx = position_in_tree[0]
    so_info = '' 
    safe_cl = ['<START>'] + cl + ['<END>']
    idx_now = head_idx + 1
    finish = False
    while not finish:
        # we traverse through the sentence in reverse order...
        idx_prev = idx_now - 1
        idx_next = idx_now + 1
        phrase_prev = safe_cl[idx_prev]
        phrase_next = safe_cl[idx_next]
        if phrase_prev == 'PP':
            idx_now = idx_now - 2
        else :
            if phrase_prev == 'VP':
                so_info = 'O'
                finish = True
            elif phrase_next == 'VP':
                so_info = 'S'
                finish = True
            elif phrase_prev == 'CC':
                so_info = 'S'
                # conjunctions indicate that
                # the NP we are examining is the subject of Sbar.
                finish = True
            elif (not phrase_prev[0].isalpha()):
                # punctuations also work similarly to conjuctions.
                so_info = 'S'
                finish = True
            elif phrase_prev != 'NP':
                idx_now = idx_now - 1
                # adverbs, unfetched adjectives, ... shall be ignored.
            else :
                so_info = 'O'   # ...(*)
                # In terms of unknown. Since subjects are more rare,
                # by returning 'O' we are more likely to have a tie and proceed
                # to the next phase.
                finish = True

    return so_info

def subject_number_info(position_in_tree, parsed_tree):
    '''
        this function gives you the two information:
        if the word token to the position_in_tree is in the subject phrase,
        and if so then whether or not that subject phrase is singular of plural.
    '''
    ppl = phrases_position_list(parsed_tree)
    cl = []
    head_idx = position_in_tree[0]
    for tup in ppl:
        if type(parsed_tree[tup]) == nltk.tree.Tree:
            cl.append(parsed_tree[tup].label())
        else :
            cl.append(parsed_tree[tup][0])
            
    if 'VP' in cl:
        vp_idx = cl.index('VP')
    else :
        return 'U'

    if head_idx >= vp_idx:
        return 'sg' # assuming that singluar verbs tend to be alone than being
                    # conjuncted with other nouns to form a plural noun phrase 

    if 'and' in cl[:vp_idx]:
        return 'pl'
    else :
        return 'sg'   

def detect_strong_relation_in_sentence(parsed_tree, pronoun_idx, target_idx):
    '''
        if pronoun and target are in the same sentence, pronoun should reference subject
        because a sentence exists to describe a subject, unless there is a significant
        indication that the pronoun references another part of the subject, for
        example being inside a PP describing an NP which is not a subject.
    '''        
    ppl = phrases_position_list(parsed_tree)
    cl = constituent_list(parsed_tree, ppl)

    if target_idx > pronoun_idx:
        intermediate = [const for const in cl[pronoun_idx : target_idx] \
                        if const[0].isalpha()]
        if set(intermediate) == set(['NP']):
            return True
        else:
            return False
    #else

    intermediate = [const for const in cl[target_idx : pronoun_idx] \
                   if const[0].isalpha()]
    if set(intermediate) == set(['NP', 'PP']):
        return True
    #else
    
    if 'VP' not in cl:
        return False #Something wrong with chunking : no VP!
    #else
    vpidx = cl.index('VP') #the first VP
    if target_idx < vpidx:
        # target is likely to be the subject of the given sentence.
        return True

    return False
    
def distance_between_words(parsed_forest, position1, position2):
    '''
        counts how many words apart, given two positions of words in the parsed forest.
    '''
    total_position_list = [(sent_num, tup) for sent_num in range(len(parsed_forest)) \
                           for tup in parsed_forest[sent_num].treepositions('leaves')]
    idx1 = total_position_list.index(position1)
    idx2 = total_position_list.index(position2)
    return abs(idx1 - idx2)

def count_occurence_of_entity(text, search_str):
    occur_indices = [s.start() for s in re.finditer(search_str + "[^A-Za-z]", text)]
    return len(occur_indices)

def random_binary_by_hash(string_of_int):
    '''
        returns 0 or 1, by computing the hash value of the input and checking the parity.
    '''
    # if every strategy we tried has failed to choose one between two...
    # we have no choice but to rely on pure luck,
    # although I strongly believe that this function will almost never be used.
    return hash(string_of_int) % 2
    

train_sents = conll2000.chunked_sents('train.txt', chunk_types = ['NP', 'VP', 'PP'])
test_sents = conll2000.chunked_sents('test.txt', chunk_types = ['NP', 'VP', 'PP'])

#### Training chunker using conll2000 corpus.
#### It takes about 30 minutes, so when we use if we load a pickled chunker.
"""
chunker = ConsecutiveNPChunker(train_sents)

print(chunker.evaluate(test_sents))

with open("./chunkerNVP.txt", "wb") as fout :
    pickle.dump(chunker, fout)
fout.close()

"""
with open("./chunkerNVP.txt", "rb") as fin:
    chunker = pickle.load(fin)
fin.close()


for page_ctxt in range(2) :
    # once for page-context, once for snippet-context.
    fin = open("./gap-test.tsv", 'r')
    fin.readline()
    
    page_context = bool(page_ctxt)
    
    if page_context :
        fout = open("./CS372_HW5_page_output_20160650.tsv", 'w')
    else :
        fout = open("./CS372_HW5_snippet_output_20160650.tsv", 'w')

    for line in fin.readlines():
        information = line.strip().split('\t')
        lineID = information[0]
        text = information[1]
        pronoun_str = information[2]
        pronoun_idx = int(information[3])
        A_str = information[4]
        A_idx = int(information[5])
        B_str = information[7]
        B_idx = int(information[8])
        url = information[10]

        sents_list = sent_tokenize(text)
        parsed_forest = [chunker.parse(pos_tag(sent)) for sent in sents_list]
        pronoun_pos = get_position_indices(sents_list, parsed_forest, pronoun_idx)
        A_pos = get_position_indices(sents_list, parsed_forest, A_idx)
        B_pos = get_position_indices(sents_list, parsed_forest, B_idx)

        pron_sent_idx, pron_position_in_tree = pronoun_pos
        A_sent_idx , A_position_in_tree = A_pos
        B_sent_idx , B_position_in_tree = B_pos

        A_res = False
        B_res = False

        go_to_next_phase = True
        
        if page_context:
            # phase 0, only used in page_context
            wikipedia_title = url.split('/')[-1]
            wikipedia_title = re.sub('_', ' ', wikipedia_title)
            parse_fail = False
            try:
                A_str_in_title = bool(re.findall(A_str, wikipedia_title))
                B_str_in_title = bool(re.findall(B_str, wikipedia_title))
            except:
                parse_fail = True

            if parse_fail is False:
                
                if A_str_in_title == B_str_in_title :
                    pass
                else :
                    A_res = A_str_in_title
                    B_res = B_str_in_title
                    go_to_next_phase = False

        if go_to_next_phase:
            
            # phase 1
            
            # pron_num_info = subject_number_info(pron_position_in_tree, parsed_forest[pron_sent_idx])
            # In the test set, pronouns are all singluar pronouns.
            # If A or B is a part of a plural noun, it won't be the one the pronoun is referencing.
            
            A_num_info = subject_number_info(A_position_in_tree, parsed_forest[A_sent_idx])
            B_num_info = subject_number_info(B_position_in_tree, parsed_forest[B_sent_idx])

            if A_num_info == 'pl' :
                if B_num_info == 'pl':
                    pass
                    #If somehow they are both detected as a part of a plural noun, we step into phase 2.
                else :
                    A_res = False
                    B_res = True
                    go_to_next_phase = False       
                    
            elif B_num_info == 'pl':
                # A is sg, B is pl
                A_res = True
                B_res = False
                go_to_next_phase = False
        
        if go_to_next_phase:

            #phase 2

            # Subject vs object/complement information plays a large role in coreference resolution.
            # Detailed explanation on why I think so is in the report.

            A_constituent_info = constituent_info(A_position_in_tree, parsed_forest[A_sent_idx])
            B_constituent_info = constituent_info(B_position_in_tree, parsed_forest[B_sent_idx])
            pron_constituent_info = constituent_info(pron_position_in_tree, parsed_forest[pron_sent_idx])

            if pron_constituent_info == 'U':
                pass
            elif pron_constituent_info == 'S':
                if A_constituent_info == 'S' :
                    if B_constituent_info == 'O':
                        A_res = True
                        B_res = False
                        go_to_next_phase = False
                        
                    else:
                        pass
                elif A_constituent_info == 'O' and B_constituent_info == 'S':
                    B_res = True
                    A_res = False
                    go_to_next_phase = False

            else : # pron_constituent_info == 'O'
                if A_constituent_info == 'O':
                    if B_constituent_info == 'S':
                        A_res = True
                        B_res = False
                        go_to_next_phase = False
                    else :
                        pass
                elif B_constituent_info == 'O' and A_constituent_info == 'S':
                    B_res = True
                    A_res = False
                    go_to_next_phase = False

        if go_to_next_phase:

            #phase 3

            # If the pronoun and A (or B) is in the same sentence then we might be able
            # to get some information from dependencies or semantics.
            pronoun_idx = pron_position_in_tree[0]
            A_target_idx = A_position_in_tree[0]
            B_target_idx = B_position_in_tree[0]

            relationA = None
            relationB = None

            if pron_sent_idx == A_sent_idx:
                relationA = detect_strong_relation_in_sentence(parsed_forest[pron_sent_idx], pronoun_idx, A_target_idx)
            
            if pron_sent_idx == B_sent_idx:
                relationB = detect_strong_relation_in_sentence(parsed_forest[pron_sent_idx], pronoun_idx, B_target_idx)

            if type(relationA) is bool and type(relationB) is bool:
                if relationA and relationB:
                    pass
                elif not (relationA or relationB) :
                    pass
                else :
                    A_res = relationA
                    B_res = relationB
                    go_to_next_phase = False
            elif relationA is None and relationB is None:
                pass
            else:
                # one got result, the other is still None
                if relationA is True:
                    A_res = True
                    B_res = False
                    go_to_next_phase = False
                elif relationB is True:
                    B_res = True
                    A_res = False
                    go_to_next_phase = False

        if go_to_next_phase:

            # phase 4
            # pronouns are used under the assumption that the reader knows
            # what the pronoun is referencing. A more frequently-occuring 
            # entity is likely to be the main subject of the text, thus more 
            # likely to be the entity the pronoun is referencing.

            try :
                A_occurence = count_occurence_of_entity(text, A_str)
                B_occurence = count_occurence_of_entity(text, B_str)

                if A_occurence < B_occurence:
                    B_res = True
                    A_res = False
                    go_to_next_phase = False
                elif B_occurence < A_occurence :
                    A_res = True
                    B_res = False
                    go_to_next_phase = False
            except :
                pass

        if go_to_next_phase:

            # phase 5
            # compare distance between words.
            
            dist_A_pron = distance_between_words(parsed_forest, pronoun_pos, A_pos)
            dist_B_pron = distance_between_words(parsed_forest, pronoun_pos, B_pos)

            if dist_A_pron < dist_B_pron :
                A_res = True
                B_res = False
                go_to_next_phase = False
                
            elif dist_B_pron < dist_A_pron:
                B_res = True
                A_res = False
                go_to_next_phase = False

        if go_to_next_phase:

            # phase 6
            # our painstaking effort to perform coreference resolution has gone 
            # in vain. We have no choice but to flip a coin.

            line_num = ''.join([char for char in lineID if char.isdigit()])
            coin_flip = random_binary_by_hash(line_num)

            if coin_flip :
                A_res = True
            else :
                B_res = True

        fout.write(lineID + '\t' + str(A_res).upper() + '\t' + str(B_res).upper() + '\n')

    fout.close()
    fin.close()

gold_tsv = "./gap-test.tsv"
snippet_tsv = "./CS372_HW5_snippet_output_20160650.tsv"
page_tsv = "./CS372_HW5_page_output_20160650.tsv"

snippet_score = gap_scorer.run_scorer(gold_tsv, snippet_tsv)
page_score = gap_scorer.run_scorer(gold_tsv, page_tsv)

print("# Snippet context results : ")
print(snippet_score)
print()
print("# Page context results : ")
print(page_score)
