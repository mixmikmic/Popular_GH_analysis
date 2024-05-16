import nltk
from urllib import urlopen

url = "http://www.gutenberg.org/cache/epub/26275/pg26275.txt"

odyssey_str = urlopen(url).read()

type(odyssey_str)

odyssey_str[3:49]

odyssey_tokens = nltk.word_tokenize(odyssey_str.decode('utf-8'))

len(odyssey_tokens)

odyssey_text = nltk.Text(odyssey_tokens)

print odyssey_text[:8]

