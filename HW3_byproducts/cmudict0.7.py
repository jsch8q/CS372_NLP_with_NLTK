from bs4 import BeautifulSoup
from urllib import request
import re

url = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
html = re.split("[\r\n]+", request.urlopen(url).read().decode('Latin-1'))

new_hetero = []
for entries in html:
    if re.match("[A-Z]+.*\(1\)", entries):
        new_hetero.append(entries.split()[0][:-3].lower())

