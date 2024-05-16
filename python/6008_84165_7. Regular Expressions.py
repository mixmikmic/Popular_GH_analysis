import nltk
import re

alice = nltk.corpus.gutenberg.words("carroll-alice.txt")

set([word for word in alice if re.search("^new",word)])

set([word for word in alice if re.search("ful$",word)])

set([word for word in alice if re.search("^..nn..$",word)])

set([word for word in alice if re.search("^[chr]at$",word)])

set([word for word in alice if re.search("^.*nn.*$",word)])

