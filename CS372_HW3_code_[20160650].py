import nltk, re, stanza, time, praw, pickle
from bs4 import BeautifulSoup
from urllib import request
from anytree import Node, RenderTree
from wiktionaryparser import WiktionaryParser 
#version of WiktionaryParser must be exactly 0.0.97 to do monkey patching
#as specified in requirements.txt
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize, sent_tokenize
from collections import defaultdict

start = time.time()
cmucorpus = nltk.corpus.cmudict 
cmudict_dict = cmucorpus.dict()
stopword = stopwords.words('english')
wnl = nltk.WordNetLemmatizer()
wikparser = WiktionaryParser()
wikparser.RELATIONS = []
stanza.download('en')
stanza_nlp = stanza.Pipeline('en', tokenize_no_ssplit=True)

# ================================================ #
# ============     END OF HEADERS     ============ #
# ================================================ #

def tag_sent(s):
    """
        POS tagging to a sentence string using NLTK.
    """
    txt = word_tokenize(s)
    return nltk.pos_tag(txt)

def get_pars(sred, verbose = True):
    """
        crawling the title, body, and comments from fetched submissions in a subreddit
    """
    paragraphs = []
    i = 0
    print("Working on reddit post #%d" %(i + 1))
    for sub in sred:
        to_add = []
        if verbose and not ((i + 1) % 50): 
            print("Working on reddit post #%d" %(i + 1))
        #paragraphs = paragraphs + [sub.title, sub.selftext]
        to_add = to_add + [sub.title, sub.selftext]
        sub.comments.replace_more(limit = None)
        comms = []
        for comment in sub.comments.list():
            comms.append(comment.body)
        to_add += comms
        sub_url = sub.url
        add_with_url = [(par, sub_url) for par in to_add]
        paragraphs += add_with_url
        i += 1
    return paragraphs
        
def normalize_sent_lists(sent_list):
    """
        Executes the following normalization process:
        1. remove any parentheses-packed clauses, as they are additional comment-like.
        2. remove askerisks, which is used as an emphasis mark.
        3. change weird-looking aphostrophes to usual ones.
        4. remove enumeration markings at the beginning of the sentence, such as #1 or 1).
        
        Also, some reddit "sentences" starts with a lowercase letter.
        So, change the first letter of the sentence to an uppercase letter.
    """
    num_sent = len(sent_list)
    for i in range(num_sent):
        sent = sent_list[i]
        sent = re.sub(r"\(.*\)|\{.*\}|\[.*\]|\*", "", sent)
        sent = re.sub(r"“|”", '"', sent)
        sent = re.sub(r"‘|’", "'", sent)
        sent = re.sub(r"^[0-9]+[\+-\.|\)|\]|\}]+\s|^[\#]+[0-9]+[\+-\.|\)|\]|\}]*\s", "", sent)
        if len(sent) :
           sent_list[i] = sent[0].upper() + sent[1:]

def heteroFromNewCMUDict(new_cmuentries):
    """
        Brings the newest version of the CMU dictionary, version 0.7b,
        from the CMU dictionary official website.
    """
    url = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
    html = re.split("[\r\n]+", request.urlopen(url).read().decode('Latin-1'))
    new_hetero = []
    for entries in html:
        if re.match("[A-Z]", entries):
            new_cmuentries.append(entries.split()[0])
        if re.match(r"[A-Z]+.*\(1\)", entries):
            new_hetero.append(entries.split()[0][:-3].lower())
    return set(new_hetero)

def heteronym_check_from_nltk(word, new_hetero):
    """
        Finds words with two different pronunciations in cmudict.
        Not meant to be the rule to determine heteronyms, 
        but to exclude words that are not heteronyms.
    """
    # new_hetero = heteroFromNewCMUDict()
    # if len(new_cmudict_hetero[word]) < 2 :
    if not (word in new_hetero):
        return False
    if len(wn.synsets(word)) < 2 :
        return False
    return True

def heteronym_check_from_wiktionary(parsed_dict):
    """
        Once the wiktionaryparser gets results from wiktionary
        and we rearrange the results as we need,
        we can be pretty sure about whether the lexical item is a heteronym or not.
        But fetching is expensive, so this should be used as less as possible.
    """
    if len(parsed_dict) < 2:
        return False
    return True

def heteronyms_from_nltk():
    """
        Makes list of words which follows the rule of heteronym_check_from_nltk().
    """
    words = [entry.lower() for entry in new_cmuentries]
    heteronym_candidates = [word for word in words if heteronym_check_from_nltk(word, hetero7)]
    # if we do not remove stopwords, not only stopwords play a weak role in a sentence in terms of context, 
    # the result will also contain undesired results such as stressed/unstressed pronunciations of articles 'a' and 'the'.
    maybe_heteros = set(heteronym_candidates).difference(set(stopword))
    return maybe_heteros

def makeDictFromWikiWord(word):
    """
        Parse the results from the WiktionaryParser further so only useful information remains.
    """
    myDict = defaultdict(dict)
    for i in range(len(word)):
        defs = []
        pron = ''
        pos = ''
        try:
            k = 0
            while not "IPA" in word[i]['pronunciations']['text'][k]:
                k += 1
            pron = word[i]['pronunciations']['text'][k]
            IPA_pron = pron[pron.find("IPA"):]
            num_of_def_chunks = len(word[i]['definitions'])
            for j in range(num_of_def_chunks):
                def_dict = dict(word[i]['definitions'][j])
                defs = def_dict['text'][1:]
                pos = def_dict['partOfSpeech']
                if pos in myDict[IPA_pron]:
                    myDict[IPA_pron][pos] += defs
                else :
                    myDict[IPA_pron][pos] = defs
        except:
            #some entries such as 'obsolete' usage of words have no pronunciations annotated.
            pass
    return dict(myDict)

def myFreq(word_list):
    """
        make information which can be used in sorting the results.
    """
    uniques = list(set(word_list))
    freq_list = []
    for word in uniques:
        freq_list.append((word_list.count(word), word))
    return freq_list        

def make_dependency_tree(stanza_tagged_sent):
    """
        After dependency parsing, this function generates a dependency tree
        using the AnyTree module.
    """
    words = stanza_tagged_sent.words
    remaining_indices = [i for i in range(len(words))]
    avoid_duplicate = 'unassinged'
    node_list = [Node('root')] + [Node(avoid_duplicate)] * len(words)
    for word in words:
        while len(remaining_indices) > 0:
            for i in remaining_indices:
                word = words[i]
                if node_list[int(word.head)].name != avoid_duplicate:
                    remaining_indices.remove(i)
                    node_list[int(word.id)] = Node((word.text, word.id, word.xpos), \
                                         parent = node_list[int(word.head)])
    return node_list

def get_helper_words_from_tree(dep_tree, node):
    """
        from the dependency tree, given a node, this function searches in the following order
        1. descendants, in order of depth
        2. if unsuccessful in step 1, parent
        3. if unsuccessful in step 2, siblings and their descendants, in order of depth
        for words which are strongly related to the word corresponding to the input node 
    """
    res_idx = []
    descendants = [child for child in node.children]
    while len(res_idx) == 0 and len(descendants) > 0:
        grandchilds = []
        for child in descendants:
            if child.name[2][0] in 'NVJ':
                res_idx.append(child.name[1])
            grandchilds += list(child.children)
        descendants = grandchilds
    if len(res_idx) > 0:
        return res_idx
    #else : look for parent
    parent_node = node.parent
    if parent_node.name != 'root':
        return [parent_node.name[1]]
    #else : parent is root, search for siblings' subtrees, i.e. the rest of the tree
    descendants = [sibling for sibling in node.siblings]
    while len(res_idx) == 0 and len(descendants) > 0:
        grandchilds = []
        for child in descendants:
            if child.name[2][0] in 'NVJ':
                res_idx.append(child.name[1])
            grandchilds += list(child.children)
        descendants = grandchilds
    return res_idx
    

def estimate_str_similarity(def_str, helper_word):
    """
        finds the most path-related pair of synsets, 
        one from the word in the input string, and the other from the input word, 
        and returns the similarity.
    """
    pos_tagged_def_str = tag_sent(def_str)
    target_pos = 'n' if helper_word.xpos[0] == 'N' else ('v' if helper_word.xpos[0] == 'V' else 'a')
    helper_word_bag = [synset for synset in wn.synsets(wnl.lemmatize(helper_word.text, target_pos))\
                                if synset.pos() == target_pos]
    maximum_similarity = 0.0
    for tagged_word, pos in pos_tagged_def_str:
        if not pos[0] in {'N', 'V', 'J'}:
            continue
        synset_bag = wn.synsets(tagged_word)
        for synset in synset_bag:
            if synset.pos() == target_pos:
                for word in helper_word_bag:
                    tmp_similarity = wn.path_similarity(word, synset)
                    if tmp_similarity is None:
                        tmp_similarity = -1
                    if tmp_similarity > maximum_similarity :
                        maximum_similarity = tmp_similarity
    
    return maximum_similarity

def estimate_list_similarity(def_pron_list, helper_word):
    #helper_word is stanza-word
    """
        does estimate_str_similarity for each string in the input list with the input word
        and returns the "scores" from each estimate_str_similarity.
    """
    def_list = [def_str for (def_str, pron, pos) in def_pron_list]
    normalize_sent_lists(def_list)
    scores = [0.0] * len(def_list)
    for i in range(len(def_list)):
        #estimate_str_similarity
        scores[i] = estimate_str_similarity(def_list[i], helper_word)
    return scores

def infer_pronunciation(def_pron_list, helper_word):
    """
        from the scores of estimate_list_similarity, 
        finds the best matching definition string with the word
        and from that definition string, infer the pronunciation of the word
    """
    pron_list = [pron for (def_str, pron, pos) in def_pron_list]
    score_list = estimate_list_similarity(def_pron_list, helper_word)
    idx = 0
    highscore = score_list[0]
    for i in range(len(score_list)):
        if score_list[i] > highscore:
            highscore = score_list[i]
            idx = i
    return (pron_list[idx], highscore, def_pron_list[idx][2] )

def reverse_dict(heterodict_entry, target_pos):
    #target_pos here should be full-name pos
    """
        a method which changes the order of heteronym dictionary entry hierarchy, 
        similar to making a reverse dictionary from a dictionary.
    """
    def_pron_list = []
    for IPA in heterodict_entry:
        defs = heterodict_entry[IPA]
        for pos in defs:
            if target_pos in pos:
                work_list = defs[pos]
                for deftext in work_list:
                    def_pron_list.append((deftext, IPA, pos))
    return def_pron_list


def determinable_by_simple_pos(word, xpos):
    """
        check : is uniquely annotating the pronunciation by POS information possible?
    """
    if xpos[0] == 'N':
        simple_pos = 'noun'
    elif xpos[0] == 'V':
        simple_pos = 'verb'
    elif xpos[0] == 'J':
        simple_pos = 'adjective'
    elif xpos[0] == 'R':
        simple_pos = 'adverb'
    else :
        simple_pos = 'etc'

    possible_prons = []
    for IPAs in heterodict[word]:
        IPA_dict = heterodict[word][IPAs]
        for pos_in_dict in IPA_dict:
            if simple_pos == pos_in_dict:
                possible_prons.append(IPAs)
    
    if len(possible_prons) == 1:
        return (True, possible_prons[0])
    else :
        return (False, None)

def determinable_by_tense_pos(word, xpos):
    """
        check : is uniquely annotating the pronunciation by POS information possible...
                ...if we add inflection information?
    """
    if xpos[0] != 'V':
        return (False, None)
    #else:
    target_pattern = 'base form'
    target_pattern_another = 'base form'

    if xpos == 'VBD':
        target_pattern = 'past tense'
    elif xpos == 'VBG':
        target_pattern = 'present participle'
    elif xpos == 'VBN':
        target_pattern = 'past participle'
    elif xpos == 'VBZ':
        target_pattern = 'Third-person singular simple present'

    if xpos not in {'VB', 'VBP'}:
        target_pattern_another = 'inflection'

    possible_prons = []
    for IPAs in heterodict[word]:
        IPA_dict = heterodict[word][IPAs]
        for pos_in_dict in IPA_dict:
            definitions = IPA_dict[pos_in_dict]
            for definition in definitions:
                if target_pattern.lower() in definition.lower() \
                   or target_pattern_another.lower() in definition.lower() :
                    possible_prons.append(IPAs)
    if len(possible_prons) == 1:
        return (True, possible_prons[0])
    else :
        return (False, None)

########################  Monkey Patching the wiktionaryparser module ########################
###### The wiktionaryparser module has a bug of not parsing the pronunciation properly, ######
####### and there are some functionalities we don't need but are called unnecessarily. #######
############## The following codes are internal methods defined in the module. ###############

def debugged_parse_pronunciation(self, word_contents):
    """
        This code fragment is included so that we can do monkey patching, 
        not use somewhere else in the code.
    """
    pronunciation_id_list = self.get_id_list(word_contents, 'pronunciation')
    pronunciation_list = []
    audio_links = []
    #pronunciation_text = []  #... in the source code this line should not be here...
    pronunciation_div_classes = ['mw-collapsible', 'vsSwitcher']
    for pronunciation_index, pronunciation_id, _ in pronunciation_id_list:
        pronunciation_text = [] #... but actually here to work properly.
        span_tag = self.soup.find_all('span', {'id': pronunciation_id})[0]
        list_tag = span_tag.parent
        while list_tag.name != 'ul':
            list_tag = list_tag.find_next_sibling()
            if list_tag.name == 'p':
                pronunciation_text.append(list_tag.text)
                break
            if list_tag.name == 'div' and any(_ in pronunciation_div_classes for _ in list_tag['class']):
                break
        for super_tag in list_tag.find_all('sup'):
            super_tag.clear()
        for list_element in list_tag.find_all('li'):
            for audio_tag in list_element.find_all('div', {'class': 'mediaContainer'}):
                #audio_links.append(audio_tag.find('source')['src'])
                audio_tag.extract()
            for nested_list_element in list_element.find_all('ul'):
                nested_list_element.extract()
            if list_element.text and not list_element.find('table', {'class': 'audiotable'}):
                pronunciation_text.append(list_element.text.strip())
        pronunciation_list.append((pronunciation_index, pronunciation_text, audio_links))
    return pronunciation_list

WiktionaryParser.parse_pronunciations = debugged_parse_pronunciation

new_cmuentries = []
hetero7 = heteroFromNewCMUDict(new_cmuentries)
# using cmudict and Wordnet, remove words those are definitely not heteronyms
hetero_candidates = heteronyms_from_nltk()

# print("preamble execution time = %.6f seconds" %(time.time() - start))

"""
###### The code block below is the part where reddit posts from wordplay subreddit is cralwed.
###### It is commented because it takes a long time to crawl reddit posts, 
###### and instead of crawling everytime, we make a pickle of crawled sentence once and for all,
###### so at the next time we need the sentences we can just load that pickle.
###### It takes about 6~8 minutes on this crawling part on my environment, 
###### but one can uncomment this part if needed. 

###########    Initializer of Python Reddit API Wrapper    ###########
### Below is a routine of authenticating via OAuth2, do not change ###

reddit = praw.Reddit(client_id = "ssLUMowJL-2Ulw", \
                     client_secret = None, \
                     redirect_uri='http://localhost:8080',
                     user_agent='jsch89')

subreddit = reddit.subreddit('wordplay')

num_of_hot_posts = 1
num_of_top_posts = 1000

### Get posts in the subreddit, sorted by hot and top ###
# hot_sred = subreddit.hot(limit = num_of_hot_posts)
top_sred = subreddit.top('all', limit = None)

pars1 = []#get_pars(hot_sred)
pars2 = get_pars(top_sred)

pars = pars1 + pars2
sents = []
for par, url in pars:
    splits = re.split("[\r\n]+", par)
    for splinter in splits:
        sent_list = sent_tokenize(splinter)
        normalize_sent_lists(sent_list)
        sent_list_with_url = [(sent, url) for sent in sent_list]
        sents += sent_list_with_url

with open("./sents_from_reddit.txt", 'wb') as fout:
    pickle.dump(sents, fout)
fout.close()

"""

# loading crawled sentences from pickle
with open("./sents_from_reddit.txt", 'rb') as fin:
    sents = pickle.load(fin)
fin.close()

# collect all words with a possibility of being a heteronym according to NLTK
pool = []
for sent, url in sents:
    sent = sent.strip()
    words = word_tokenize(sent)
    weak_heteros = []
    cnt = 0
    for word in words:
        if word.lower() in hetero_candidates:
            cnt += 1
            weak_heteros.append(word.lower())
    if cnt :
        pool = pool + weak_heteros

"""
###### The code block below is the part where we query Wiktionary to check if a word is really a heteronym.
###### It is commented because it takes a long time due to crawling and parsing many times. 
###### So instead of doing this everytime, we make a pickle of dictionary conatining fetched results,
###### so at the next time we need the sentences we can just load that pickle.
###### It takes about 15~18 minutes on ~900 queries on my environment, 
###### but this part heavily depends on the input courpus, so if the inpurt corpus is changed
###### then this part must be uncommented and executed. 

start2 = time.time()
new_pool = []
hetero_dict = {}
for word in set(pool):
    tmp_dict = makeDictFromWikiWord(wikparser.fetch(word))
    if heteronym_check_from_wiktionary(tmp_dict):
        hetero_dict[word] = tmp_dict
        new_pool.append(word)

with open("./heteronym_pickle.txt", 'wb') as fout:
    pickle.dump(hetero_dict, fout)
fout.close()
"""

heterodict = {}
with open("./heteronym_pickle.txt", 'rb') as fin:
    heterodict = pickle.load(fin)
fin.close()

# The annotating part
start3 = time.time()
sent_count = []
for sent, url in sents:
    heteros_in_sent = []
    to_analyze = []
    words_list = word_tokenize(sent)
    # collect all indices of heteronyms in sentence
    for i in range(len(words_list)):
        word = words_list[i].lower()
        if word in heterodict:
            heteros_in_sent.append(word)
            to_analyze.append(i)
    if len(heteros_in_sent):
        # start annotation only if there is a heteronym in the sentence
        sent_to_doc = stanza_nlp(sent)
        tagged_sent = sent_to_doc.sentences[0]
        # map to store annotation results
        annotation_dict = {}
        for word_idx in to_analyze: 
            word_info_from_stanza = tagged_sent.words[word_idx]
            word_to_lookup = words_list[word_idx].lower()
            if not word_to_lookup in heterodict:
                raise ValueError
            word_pos = word_info_from_stanza.xpos
            full_name_pos = 'noun' if word_pos[0] == 'N' else ('verb' if word_pos[0] == 'V'\
                                         else ('adjective' if word_pos[0] == 'J' else ('adverb' if word_pos[0] == 'R' else '')))
            
            # check if knowing POS information, and/or how the word is inflected can uniquely determine the pronunciation
            tag_done, IPA_tag = determinable_by_simple_pos(word_to_lookup, word_pos)
            if tag_done :
                annotation_dict[(word_to_lookup, word_idx)] = (IPA_tag, full_name_pos)
                continue
            tag_done, IPA_tag = determinable_by_tense_pos(word_to_lookup, word_pos)
            if tag_done:
                annotation_dict[(word_to_lookup, word_idx)] = (IPA_tag, full_name_pos)
                continue

            # no early detection, dependecy parsing and analysis should be done.
            sent_dep_tree = make_dependency_tree(tagged_sent)
            helpers_indices = get_helper_words_from_tree(sent_dep_tree,  sent_dep_tree[int(word_info_from_stanza.id)] )
            def_pron_list = reverse_dict(heterodict[word_to_lookup], full_name_pos)
            if len(def_pron_list) == 0:
                #this case, pos_tagger tagged a POS not listed in wiktionary. We should retry without using POS information.
                def_pron_list = reverse_dict(heterodict[word_to_lookup], '')

            inference_result = [] # estimation scores will be stored in this list.
            for helpers_index in helpers_indices:
                IPA_tag, likely, infered_pos = infer_pronunciation(def_pron_list, tagged_sent.words[int(helpers_index) - 1])
                inference_result.append((IPA_tag, likely, infered_pos))

            if len(inference_result) == 0:
                # In this case, no useful related words to the heteronym was found in the sentence...
                # which in other words, does not matter much or cannot determine decisively how we read this heteronym;
                # for example, the sentence "Read it." can be a response to a question "Did you read?", or an impertative sentence.
                IPA_tag = list(heterodict[word_to_lookup])[0]
                annotation_dict[(word_to_lookup, word_idx)] = ( IPA_tag , list(heterodict[word_to_lookup][IPA_tag])[0] )
                continue

            # annotate the pronunciation with the best estimation score 
            final_IPA_tag = inference_result[0][0]
            max_likely = inference_result[0][1]
            final_infered_pos = inference_result[0][2]
            for result in inference_result:
                if result[1] > max_likely:
                    max_likely = result[1]
                    final_IPA_tag = result[0]
                    final_infered_pos = result[2]
            annotation_dict[(word_to_lookup, word_idx)] = (final_IPA_tag, final_infered_pos)
        # appending information so that the sorted order follows the rules specified in the HW description document
        sent_count.append([len(heteros_in_sent), myFreq(heteros_in_sent), len(set(annotation_dict.values())), sent, annotation_dict, url]) 
    
fout = open("./CS372_HW3_output_[20160650].csv", 'w', encoding = "utf-8")

new_sent = sorted(sent_count, reverse = True) #sort the results
for (total_freq, cnt, num_of_distinct_pron, sent, annotations, url) in new_sent[:30]:
    fout.write('"' + sent + '"')
    fout.write(' , ' + str(annotations)[1:-1])
    fout.write(' , ' + url) # precise citation indicating the reddit submission URL.
    fout.write('\n')
fout.close()
#print("annotation and writing file : %.6f seconds" %(time.time() - start3))
