#Import Statements
import string
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

#Sentence object definition for data import and processing 
class Sentence:
    def __init__(self, document, sentenceNumber, wordList, lemmaList, posList):
        self.document = document
        self.sentenceNumber = sentenceNumber
        self.wordList = wordList
        self.lemmaList = lemmaList
        self.posList = posList
        self.sentence = " ".join([word for word in wordList if word not in string.punctuation])
        self.lemmaSent = " ".join([word for word in lemmaList if word not in string.punctuation])

#Get the files we want to process and put them into a list of lists called sentList
fileList = os.listdir("../PubMed/pmc_dddb_full")
sentList = []
fileList.sort()
for n in range(35, 36):
    f = open("../PubMed/pmc_dddb_full/" + fileList[n], 'r')
    for line in f:
        sentList.append(line.split('\t'))

len(sentList)

docList = []
for thing in sentList:
    docList.append(thing[0])

len(set(docList))

sentObjList = []
def processSent(item):
    wordList = item[3].replace('"',"").lstrip("{").rstrip("}").split(",")
    wordList = filter(None, wordList)
    posList = item[4].split(",")
    lemmaList = item[6].replace('"',"").lstrip("{").rstrip("}").split(",")
    lemmaList = filter(None, lemmaList)
    return Sentence(item[0], item[1], wordList, lemmaList, posList)

po = Pool(16)
results = [po.apply_async(processSent, args = (sent,)) for sent in sentList]
po.close()
po.join()
output = [p.get() for p in results]
sentObjList = output
sentObjList[7].lemmaSent

headingsDict = defaultdict(dict)

for sent in sentObjList:
    if len(sent.wordList) == 1:
        #print(sent.wordList)
        word = string.upper(sent.wordList[0]).strip()
        if word == 'INTRODUCTION' or word == 'BACKGROUND':
            headingsDict[sent.document]["introduction"] = sent.sentenceNumber
        elif word == 'METHODS':
            headingsDict[sent.document]["methods"] = sent.sentenceNumber
        elif word == 'RESULTS':
            headingsDict[sent.document]["results"] = sent.sentenceNumber
        elif word == 'DISCUSSION':
            headingsDict[sent.document]["discussion"] = sent.sentenceNumber
        elif word == 'CONCLUSION':
            headingsDict[sent.document]["conclusion"] = sent.sentenceNumber
        elif word == 'REFERENCES':
            headingsDict[sent.document]["references"] = sent.sentenceNumber
        

headingsDict.keys()

documentDict = defaultdict(list)
docPartsDict = defaultdict(lambda : defaultdict(list))
docPartsCombinedDict = defaultdict(dict)

for item in sentObjList:
    documentDict[item.document].append(item)
    
for document in documentDict.keys():
    docSentList = documentDict[document]
    introNum = int(headingsDict[document].get("introduction", -1))
    methoNum = int(headingsDict[document].get("methods", -1))
    resultNum = int(headingsDict[document].get("results", -1))
    discussNum = int(headingsDict[document].get("discussion", -1))
    conclusionNum = int(headingsDict[document].get("conclusion", -1))
    refNum = int(headingsDict[document].get("references", -1))

    for sent in docSentList:
        label = "noSection"
        dist = int(sent.sentenceNumber)
        sentNumber = int(sent.sentenceNumber)
        
        if dist > sentNumber - introNum and sentNumber - introNum > 0:
            label = "introduction"
            dist = sentNumber - introNum
        if dist > sentNumber - methoNum and sentNumber - methoNum > 0:
            label = "methods"
            dist = sentNumber - methoNum
        if dist > sentNumber - resultNum and sentNumber - resultNum > 0:
            label = "results"
            dist = sentNumber - resultNum
        if dist > sentNumber - discussNum and sentNumber - discussNum > 0:
            label = "discussion"
            dist = sentNumber - discussNum
        if dist > sentNumber - conclusionNum and sentNumber - conclusionNum > 0:
            label = "conclusion"
            dist = sentNumber - conclusionNum
        if dist > sentNumber - refNum and sentNumber - refNum > 0:
            label = "references"
            dist = sentNumber - refNum
        if sent.sentence.strip().lower() not in ["introduction", "methods", "results", "discussion", "conclusion", "references"]:
            docPartsDict[document][label].append(sent)
    
    for x in docPartsDict[document].keys():
        docPartsCombinedDict[document][x] = " ".join(" ".join(y.posList) for y in sorted(docPartsDict[document][x], key=lambda z: z.sentenceNumber))

validDocsDict = defaultdict(dict)

for doc in docPartsCombinedDict.keys():
    tempKeys = docPartsCombinedDict[doc].keys()
    if 'introduction' in tempKeys and 'discussion' in tempKeys and 'conclusion' in tempKeys:
        validDocsDict[doc] = docPartsCombinedDict[doc]

print(str(len(docPartsCombinedDict.keys())))
print(str(len(validDocsDict.keys())))

outputDict = dict()
for doc in validDocsDict.keys():
    for part in validDocsDict[doc].keys():
        outputDict[part + "_" + doc] = validDocsDict[doc][part]

pickle.dump(outputDict, open("PubmedPos35.p", "wb"))

