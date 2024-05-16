import nltk

text = "I walked to the cafe to buy coffee before work."

tokens = nltk.word_tokenize(text)

nltk.pos_tag(tokens)

nltk.help.upenn_tagset()

nltk.pos_tag(nltk.word_tokenize("I will have desert."))

nltk.pos_tag(nltk.word_tokenize("They will desert us."))

md = nltk.corpus.gutenberg.words("melville-moby_dick.txt")

md_norm = [word.lower() for word in md if word.isalpha()]

md_tags = nltk.pos_tag(md_norm,tagset="universal")

md_tags[:5]

md_nouns = [word for word in md_tags if word[1] == "NOUN"]

nouns_fd = nltk.FreqDist(md_nouns)

nouns_fd.most_common()[:10]  

