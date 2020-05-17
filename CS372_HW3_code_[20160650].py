import nltk, stanza
from bs4 import BeautifulSoup
from urllib import request
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
cmucorpus = nltk.corpus.cmudict 

cmudict_dict = cmucorpus.dict()
cmuentries = cmucorpus.entries()

def tag_sent(s):
    txt = word_tokenize(s)
    return nltk.pos_tag(txt)

def google_query(s):
    return '+'.join(s.split())

def get_google_search_urls(s):
    res_urls = []
    url = google_search + google_query(s)
    header_url = request.Request(url, headers = hdr)
    html = request.urlopen(header_url).read().decode('utf8')
    raw = str(BeautifulSoup(html, 'html.parser')).split()
    href_ind = 'href="/url?q='
    magic_ending = '"><div'
    for item in raw:
        if href_ind == item[ : len(href_ind)] and magic_ending == item[-len(magic_ending) : ]:
            new_url = item.split('&amp;sa=')[0][len(href_ind) :]
            res_urls.append(new_url)
    return res_urls

def destress(pronunciation):
    res = ""
    for syllable in pronunciation:
        if syllable[-1].isdigit():
            res += syllable[:-1]
        else :
            res += syllable
    return res

def heteronym_check_from_cmu(word):
    pronunciations = cmudict_dict[word]
    number_of_ways = len(pronunciations)
    if number_of_ways < 2 :
        return False
    destressed_pronunciations = set()
    for pronunciation in pronunciations:
        destressed_pronunciation = destress(pronunciation)
        destressed_pronunciations.add(destressed_pronunciation)
    if len(destressed_pronunciations) < 2 :
        return False
    return True

def heteronyms_from_cmudict():
    words = [entry[0] for entry in cmuentries]
    heteronym_candidates = [word for word in words if heteronym_check_from_cmu(word)]
    """for word in words:
        if heteronym_check_from_cmu(word):
            heteronym_candidates.append(word)"""
    return list(set(heteronym_candidates))
"""
def bass_def(token = "bass"):
    for item in wn.synsets(token):
        if item.pos() == 'n':
            print(item.definition())

test_sent = "The bandage was wound around the wound."

stanza.download('en')
nlp = stanza.Pipeline('en')

doc = nlp(test_sent)
print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')


bass = wn.synset('sea_bass.n.01')
swim = wn.synset('swim.v.01')
fish = wn.synset('fish.n.01')
bass_guitar = wn.synset('bass.n.07')
instrument = wn.synset('musical_instrument.n.01')
play = wn.synset('play.v.03')
drum = wn.synset('drum.n.01')
"""
"""
google_search = "https://www.google.com/search?q="
hdr = {'User-Agent': 'Mozilla/5.0'}

urls1 = get_google_search_urls("heteronym pun reddit")
urls2 = get_google_search_urls("heteronym used in a sentence")

for url in urls1+urls2:
    print(url)


print(tag_sent(test_sent))
"""

hetero_candidates = heteronyms_from_cmudict()
print(len(hetero_candidates))
