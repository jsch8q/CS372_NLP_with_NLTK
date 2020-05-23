import nltk, re, stanza, time, praw, pickle
from bs4 import BeautifulSoup
from urllib import request
from wiktionaryparser import WiktionaryParser
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize, sent_tokenize
from pprint import pprint
from collections import defaultdict

start = time.time()
# cmucorpus = nltk.corpus.cmudict 
# cmudict_dict = cmucorpus.dict()
# cmuentries = cmucorpus.entries()
# words_in_cmu = set(cmudict_dict)
stopword = stopwords.words('english')
wnl = nltk.WordNetLemmatizer()
wikparser = WiktionaryParser()
wikparser.RELATIONS = []
#stanza.download('en')
#stanza_nlp = stanza.Pipeline('en')

# ================================================ #
# ============     END OF HEADERS     ============ #
# ================================================ #

def tag_sent(s):
    """
        POS tagging to a sentence string.
        Since we are using stanza, maybe this should be replaced by that.
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
        if verbose and not ((i + 1) % 50): 
            print("Working on reddit post #%d" %(i + 1))
        paragraphs = paragraphs + [sub.title, sub.selftext]
        sub.comments.replace_more(limit = None)
        comms = []
        for comment in sub.comments.list():
            comms.append(comment.body)
        paragraphs += comms
        i += 1
    return paragraphs
        
def normalize_sent_lists(sent_list):
    """
        Some reddit "sentences" starts with a lowercase letter.
        Changes the first letter of the sentence to an uppercase letter.
        But is this process necessary?
    """
    num_sent = len(sent_list)
    for i in range(num_sent):
        sent = sent_list[i]
        sent = re.sub("\(.*\)|\{.*\}|\[.*\]|\*", "", sent)
        if len(sent) :
           sent_list[i] = sent[0].upper() + sent[1:]

def heteroFromNewCMUDict(new_cmuentries):
    url = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
    html = re.split("[\r\n]+", request.urlopen(url).read().decode('Latin-1'))
    new_hetero = []
    for entries in html:
        if re.match("[A-Z]", entries):
            new_cmuentries.append(entries.split()[0])
        if re.match("[A-Z]+.*\(1\)", entries):
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
        Once the wiktionaryparser gets results from wiktionary,
        we can be pretty sure about whether the lexical item is a heteronym or not.
        But fetching is expensive, so this should be used as less as possible.
    """
    if len(parsed_dict) < 2:
        return False
    return True
    if len(word) < 2:
        # this case, the word has only one etymology, 
        # thus only one pronunciation is assigned to this lexical item.
        return False
    pron_set = set()
    for i in range(len(word)):
        if not '(obsolete)' in wd[i]['definitions']:
            try :
                pron_set.add(word[i]['pronunciations']['text'][0])
            except :
                #some 'obsolete' usage of words have no pronunciations annotated.
                pass
    #pron_set = set([word[i]['pronunciations']['text'][0] for i in range(len(word)) if not '(obsolete)' in wd[i]['definitions']])
    if len(pron_set) < 2:
        return False
    # FIXME : There are words having multiple etymologies but a single pronunciation.
    #       : Thus, we shall not just blindly return True.
    # I think this is FIXED.
    return True

def heteronyms_from_nltk():
    """
        Makes list of words which follows the rule of heteronym_check_from_nltk().
    """
    words = [entry.lower() for entry in new_cmuentries]
    heteronym_candidates = [word for word in words if heteronym_check_from_nltk(word, hetero7)]
    maybe_heteros = set(heteronym_candidates).difference(set(stopword))
    return maybe_heteros

def makeDictFromWikiWord(word):
    myDict = defaultdict(dict)
    for i in range(len(word)):
        defs = []
        pron = ''
        pos = ''
        try:
            pron = word[i]['pronunciations']['text'][0]
            IPA_pron = pron[pron.find("IPA"):]
            num_of_def_chunks = len(word[i]['definitions'])
            for j in range(num_of_def_chunks):
                def_dict = dict(word[i]['definitions'][j])
                defs = def_dict['text'][1:]
                pos = def_dict['partOfSpeech']
                myDict[IPA_pron][pos] = defs
        except:
            #some entries such as 'obsolete' usage of words have no pronunciations annotated.
            pass
    return dict(myDict)

def myFreq(word_list):
    uniques = list(set(word_list))
    freq_list = []
    for word in uniques:
        freq_list.append((word_list.count(word), word))
    return freq_list        

########################  Monkey Patching the wiktionaryparser module ########################
###### The wiktionaryparser module has a bug of not parsing the pronunciation properly. ######
############## The following code is an internal method defined in the module. ###############

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
                audio_links.append(audio_tag.find('source')['src'])
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
hetero_candidates = heteronyms_from_nltk()

"""
###    Initializer of Python Reddit API Wrapper    ###
### Below is a routine of authenticating via OAuth2 ##
reddit = praw.Reddit(client_id = "ssLUMowJL-2Ulw", \
                     client_secret = None, \
                     redirect_uri='http://localhost:8080',
                     user_agent='jsch89')

subreddit = reddit.subreddit('wordplay')



# word = wikparser.fetch("a")
# pprint(word)

#num_of_hot_posts = 0
#num_of_top_posts = 1000

### Get posts in the subreddit, sorted by hot and top ###
# hot_sred = subreddit.hot(limit = num_of_hot_posts)
top_sred = subreddit.top('all', limit = None)

pars1 = []#get_pars(hot_sred)
pars2 = get_pars(top_sred)

pars = pars1 + pars2
sents = []
for par in pars:
    splits = re.split("[\r\n]+", par)
    for splinter in splits:
        sent_list = sent_tokenize(splinter)
        normalize_sent_lists(sent_list)
        sents += sent_list

with open("./sents_from_reddit.txt", 'wb') as fout:
    pickle.dump(sents, fout)
fout.close()

print("crawled %d sentences from %d submissions in %.6f seconds" %( len(sents), (num_of_top_posts if bool(num_of_top_posts) else 1000) + (num_of_hot_posts if bool(num_of_hot_posts) else 1000), time.time() - start))
"""

with open("./sents_from_reddit.txt", 'rb') as fin:
    sents = pickle.load(fin)

fin.close()
#print(sents[:5])
#_ = input("cry cry")

fout = open("./reddit.txt", 'w', encoding = "utf-8")
sent_count = []
for sent in sents:
    sent = sent.strip()
    words = word_tokenize(sent)
    weak_heteros = []
    cnt = 0
    for word in words:
        if word.lower() in hetero_candidates:
            cnt += 1
            weak_heteros.append(word.lower())
    if cnt >= 2:
        #do something
        sent_count.append([myFreq(weak_heteros), sent])

new_sent = sorted(sent_count, reverse = True)
for (cnt, sent) in new_sent:
    fout.write(sent)
    fout.write(' : ' + str(cnt))
    fout.write('\n')
fout.close()


############WIKTIONARY_RELATED_TEST############
if input("TEST? Y/n : ") == "Y":
    noword = ['zero', 'one', 'two', 'get', 'watch', 'eleven', 'good', 'job']
    yesword = ['tear', 'bow', 'produce', 'wind', 'ellipses', 'bass', 'does', "dove"]
    for i in range(1):
            stt = time.time()
            for word in noword:
                    wd = wikparser.fetch(word)
                    wdict = makeDictFromWikiWord(wd)
                    print(heteronym_check_from_wiktionary(wdict))
            print(time.time() - stt)
    for i in range(1):
            stt = time.time()
            for word in yesword:
                    wd = wikparser.fetch(word)
                    wdict = makeDictFromWikiWord(wd)
                    print(heteronym_check_from_wiktionary(wdict))
            print(time.time() - stt)
