import nltk

alice = nltk.corpus.gutenberg.words("carroll-alice.txt")

alice = alice[:1000]

alice_str = ' '.join(alice)

alice_str

new_file = open('export.txt', 'w')

new_file.write(alice_str)

new_file.close()

