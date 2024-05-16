import nltk

alice = nltk.corpus.gutenberg.words("carroll-alice.txt")

alice_fd = nltk.FreqDist(alice)

alice_fd

alice_fd["Rabbit"]

alice_fd.most_common(15)

alice_fd.hapaxes()[:15]

