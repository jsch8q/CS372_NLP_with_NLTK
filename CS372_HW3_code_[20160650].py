import praw, time, re
import nltk, re, stanza, time, praw
from bs4 import BeautifulSoup
from urllib import request
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize, sent_tokenize

start = time.time()
cmucorpus = nltk.corpus.cmudict 
cmudict_dict = cmucorpus.dict()
cmuentries = cmucorpus.entries()
words_in_cmu = set(cmudict_dict)
stopword = stopwords.words('english')
wnl = nltk.WordNetLemmatizer()

def get_pars(sred):
    """
        crawling the title, body, and comments from fetched submissions in a subreddit
    """
    paragraphs = []
    i = 0
    print("Working on submission #%d" %(i + 1))
    for sub in sred:
        if not ((i + 1) % 50): 
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

reddit = praw.Reddit(client_id = "ssLUMowJL-2Ulw", \
                     client_secret = None, \
                     redirect_uri='http://localhost:8080',
                     user_agent='jsch89')

subreddit = reddit.subreddit('wordplay')

num_of_submissions = 250

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
      
