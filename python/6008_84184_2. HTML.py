import nltk
from urllib import urlopen

url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

html = urlopen(url).read()

html

from bs4 import BeautifulSoup

web_str = BeautifulSoup(html).get_text()

web_tokens = nltk.word_tokenize(web_str)

web_tokens[0:25]

start = web_str.find("Python is a widely used general-purpose, high-level programming language.")

end = web_str.find("CPython is managed by the non-profit Python Software Foundation.")

last_sent = len("CPython is managed by the non-profit Python Software Foundation.")

intro = web_str[start:end+last_sent]

intro_tokens = nltk.word_tokenize(intro)

print intro_tokens

