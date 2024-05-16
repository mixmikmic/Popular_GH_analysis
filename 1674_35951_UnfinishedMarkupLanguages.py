import urllib.request
aliceUrl = "http://www.gutenberg.org/files/11/11-h/11-h.htm"
aliceString = urllib.request.urlopen(aliceUrl).read()

print(aliceString[:200]," ... ",aliceString[-200:]) # have a peek

from bs4 import BeautifulSoup
aliceSoup = BeautifulSoup(aliceString)
aliceSoupText = aliceSoup.text # this includes the head section
print(aliceSoupText[:200]," ... ",aliceSoupText[-200:]) # have a peek

aliceSoupBodyText = aliceSoup.body.text
print(aliceSoupBodyText[:200]," ... ",aliceSoupBodyText[-200:]) # have a peek

import nltk
rjFile = nltk.data.find("corpora/shakespeare/r_and_j.xml") # search for this text
rjFile

from xml.etree.ElementTree import ElementTree
rjXml = ElementTree().parse(rjFile) # parse the file
rjXml

speeches = rjXml.findall(".//SPEECH")
len(speeches)

speakers = rjXml.findall(".//SPEAKER")
len(speeches) # same thing, each speech has a speaker tag

speakerNames = [speaker.text for speaker in speakers]

print(set(speakerNames)) # unique speakers

get_ipython().magic('matplotlib inline')
nltk.FreqDist(speakerNames).plot(20) # how many speeches for each speaker

uniqueSpeakerNames = list(set(speakerNames))
print(uniqueSpeakerNames)
# let's look at names that aren't the non-string None and aren't all uppercase
titleCaseSpeakerNames = [name for name in uniqueSpeakerNames if name != None and name != name.upper()]
nltk.Text(speakerNames).dispersion_plot(titleCaseSpeakerNames)

# let's create a dictionary with each speaker pointing to text from that speaker
speakersDict = nltk.defaultdict(str)
speeches = rjXml.findall(".//SPEECH")
for speech in speeches:
    speaker = speech.find("SPEAKER").text
    for line in speech.findall("LINE"):
        if line.text:
            speakersDict[speaker]+=line.text+"\n"

# now let's look at speech length for each speaker (different from number of speeches)
speakersLengthsDict = {}
for speaker, text in speakersDict.items():
    speakersLengthsDict[speaker]=len(text)

nltk.FreqDist(speakersLengthsDict).plot(20)

# let's look at how often Romeo and Juliet say "love" and which words are nearby
romeoTokens = nltk.word_tokenize(speakersDict["ROMEO"])
print(romeoTokens.count("love")/len(romeoTokens))
nltk.Text(romeoTokens).similar("love")
julietTokens = nltk.word_tokenize(speakersDict["JULIET"])
print(julietTokens.count("love")/len(julietTokens))
nltk.Text(julietTokens).similar("love")



