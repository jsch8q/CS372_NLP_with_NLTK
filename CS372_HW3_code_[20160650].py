import nltk, re, stanza, time, praw
from bs4 import BeautifulSoup
from urllib import request
from wiktionaryparser import WiktionaryParser
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize, sent_tokenize
from pprint import pprint

start = time.time()
cmucorpus = nltk.corpus.cmudict 
cmudict_dict = cmucorpus.dict()
cmuentries = cmucorpus.entries()
words_in_cmu = set(cmudict_dict)
stopword = stopwords.words('english')
wnl = nltk.WordNetLemmatizer()
wikparser = WiktionaryParser()
wikparser.RELATIONS = []
#stanza.download('en')
#stanza_nlp = stanza.Pipeline('en')

# ================================================#
# ============     END OF HEADERS     ============#
# ================================================#

def tag_sent(s):
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
            print("Working on submission #%d" %(i + 1))
        paragraphs = paragraphs + [sub.title, sub.selftext]
        sub.comments.replace_more(limit = None)
        comms = []
        for comment in sub.comments.list():
            comms.append(comment.body)
        paragraphs += comms
        i += 1
    return paragraphs
        
def normalize_sent_lists(sent_list):
    num_sent = len(sent_list)
    for i in range(num_sent):
        sent = sent_list[i]
        if len(sent) :
           sent_list[i] = sent[0].upper() + sent[1:]

def heteronym_check_from_cmu(word):
    if len(cmudict_dict[word]) < 2 :
        return False
    return True

def heteronyms_from_cmudict():
    words = [entry[0] for entry in cmuentries]
    heteronym_candidates = [word for word in words if heteronym_check_from_cmu(word)]
    maybe_heteros = set(heteronym_candidates).difference(set(stopword))
    return maybe_heteros

########################  Monkey Patching the wiktionaryparser module ########################
###### The wiktionaryparser module has a bug of not parsing the pronunciation properly. ######
############## The following code is an internal method defined in the module. ###############

def debugged_parse_pronunciation(self, word_contents):
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


###    Initializer of Python Reddit API Wrapper    ###
### Below is a routine of authenticating via OAuth2 ##
reddit = praw.Reddit(client_id = "ssLUMowJL-2Ulw", \
                     client_secret = None, \
                     redirect_uri='http://localhost:8080',
                     user_agent='jsch89')

subreddit = reddit.subreddit('wordplay')

hetero_candidates = heteronyms_from_cmudict()

word = wikparser.fetch("a")
pprint(word)

"""num_of_submissions = 50 #250

### Get posts in the subreddit, sorted by hot and top ###
hot_sred = subreddit.hot(limit = num_of_submissions)
top_sred = subreddit.top('all', limit = num_of_submissions)

pars1 = get_pars(hot_sred)
pars2 = get_pars(top_sred)

pars = pars1 + pars2
sents = []
for par in pars:
    splits = re.split("[\r\n]+", par)
    for splinter in splits:
        sent_list = sent_tokenize(splinter)
        normalize_sent_lists(sent_list)
        sents += sent_list

print("crawled %d sentences from %d submissions in %.6f seconds" %( len(sents), (num_of_submissions if bool(num_of_submissions) else 1000) * 2, time.time() - start))
"""
