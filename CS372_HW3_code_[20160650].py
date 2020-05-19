import nltk, re, stanza , time
from bs4 import BeautifulSoup
from urllib import request
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize, sent_tokenize
cmucorpus = nltk.corpus.cmudict 

cmudict_dict = cmucorpus.dict()
cmuentries = cmucorpus.entries()
stopword = stopwords.words('english')

def tag_sent(s):
    txt = word_tokenize(s)
    return nltk.pos_tag(txt)

def google_query(s):
    return '+'.join(s.split())

def make_soup(url):
    header_url = request.Request(url, headers = hdr)
    try:
        html = request.urlopen(header_url).read().decode('utf8')
    except(UnicodeDecodeError):
        try :
            html = request.urlopen(header_url).read().decode('Latin-1')
        except:
            html = request.urlopen(request.Request("https://www.bbc.com/", headers = hdr)).read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')
    return soup

def get_google_search_urls(s):
    res_urls = []
    url = google_search + google_query(s)
    soup = make_soup(url)
    # header_url = request.Request(url, headers = hdr)
    # html = request.urlopen(header_url).read().decode('utf8')
    # raw = str(BeautifulSoup(html, 'html.parser')).split()
    raw = str(soup).split()
    href_ind = 'href="/url?q='
    magic_ending = '"><div'
    for item in raw:
        if href_ind == item[ : len(href_ind)] \
            and magic_ending == item[-len(magic_ending) : ]:
            new_url = item.split('&amp;sa=')[0][len(href_ind) :]
            res_urls.append(new_url)
    return res_urls

def str_starts_with_enumeration(s):
    return bool(re.match('^[0-9]+[^A-Za-z]', s))

def split_sent_by_colon(s):
    # s = s.strip()
    new_sents = re.split(";|:", s)
    for i in range(len(new_sents)):
        sent = new_sents[i]
        new_sent = sent.strip()
        if len(new_sent) == 0:
            new_sents[i] = "Bye."
        # elif len(new_sent) == 1:
        #     new_sents[i] = new_sent.upper()
        else :
            new_sents[i] = new_sent[0].upper() + new_sent[1:]
    return new_sents

def get_paragraphs_from_url(url):
    soup = make_soup(url)
    raw = soup.find_all("p")
    paragraph_list = []
    for item in raw:
        sents =  [sent for sent in re.split(r'\r|\n', item.text) if sent != '']
        paragraph_list += sents
    paragraph_list = list(set(paragraph_list))
    # res = paragraph_list
    res = []
    for paragraph in paragraph_list:
        res = res + split_sent_by_colon(paragraph)
    refined_res = refine_crawled_result_further(res)
    return refined_res

def is_one_word(sent):
    return not bool(len(sent.split()) - 1)

def has_curly_braces(sent):
    return True if ( '{' in sent or '}' in sent ) else False

def refine_crawled_result_further(res_list):
    refined_res = []
    for item in res_list:
        item = item.strip()

        if str_starts_with_enumeration(item):
            for i in range(len(item)):
                if not item[i].isdigit():
                    item = item[i+1:].split()
                    break

        if re.search(r'[0-9]+px', item) or re.search(r'[^A-Za-z0-9.,?!\'"(){}\[\] ]', item) or is_one_word(item) or has_curly_braces(item):
            pass
        else :
            refined_res.append(item)
    return refined_res

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

start = time.time()
google_search = "https://www.google.com/search?q="
hdr = {'User-Agent': 'Mozilla/5.0'}

queries = ["site:reddit.com heteronym example", "heteronym used in a sentence"]

urls = []
for query in queries:
    tmp_urls = get_google_search_urls(query)
    urls = urls + tmp_urls
urls = list(set(urls))

fout = open("./reddit.txt", "w", encoding = "utf-8")
urlout = open("./searched_urls.txt", "w", encoding = "utf-8")
crawled_sents = []

#urls = urls1+urls2
i = 1

print(len(urls), " urls found. Crawling Started...")
for url in urls:#[i:i+1]:
    urlout.write(urls[i-1])
    urlout.write("\n")
    print("Attempting to crawl from URL #", i)

    try:
        soup = make_soup(url)
        raw = get_paragraphs_from_url(url)
        for item in raw:
            # if True or str_starts_with_enumeration(item):
            fout.write(item)
            fout.write("\n")
            crawled_sents += sent_tokenize(item)
    except :
        pass
    i += 1

fout.close() 
urlout.close()
print("time elapsed : ", time.time() - start, "seconds")

"""
print(tag_sent(test_sent))
"""
hetero_candidates = heteronyms_from_cmudict()
hetero_candidates = set(hetero_candidates).difference(set(stopword))
print(len(hetero_candidates))


#url = "https://www.reddit.com/r/grammar/comments/24kegp/why_are_bow_and_bow_pronounced_differently/"
# contents = get_paragraphs_from_url(url)
# for item in contents:
#     print(item)


# url = "http://jonv.flystrip.com/heteronym/heteronym.htm"
# url = "http://www.fun-with-words.com/nym_heteronyms.html"

# # contents = get_paragraphs_from_url(url)
# # for sent in contents:
# #     if str_starts_with_enumeration(sent):
# #         print(sent)

# soup = make_soup(url)
# # fout = open("./reddit.txt", "w", encoding = "utf-8")
# #raw = re.split(r"\r|\n", soup.get_text())
# raw = get_paragraphs_from_url(url)
# for item in raw:
#     if True or str_starts_with_enumeration(item):
#         print(item)
#         # fout.write("\n")
# # fout.close()
