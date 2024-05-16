import nltk

names = [("Group A", "Paul"),("Group A", "Mike"),("Group A", "Katy"),("Group B", "Amy"),("Group B", "Joe"),("Group B", "Amy")]

names

nltk.FreqDist(names)

nltk.ConditionalFreqDist(names)

