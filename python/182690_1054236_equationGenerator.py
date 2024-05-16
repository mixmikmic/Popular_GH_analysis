get_ipython().system('pip install -i https://pypi.anaconda.org/pypi/simple antlr4-python2-runtime')

get_ipython().system('git clone https://github.com/augustt198/latex2sympy')

get_ipython().system('cd latex2sympy; antlr4 PS.g4 -o gen')

get_ipython().system('pip install mxnet')

import json
from sympy import *
import pprint
import re
import copy
import random
import compiler
import mxnet as mx
import numpy as np
import sys
sys.path.append("./latex2sympy")
from eqGen import readJson, readAxioms, parseEquation, buildEq, buildAxiom, genPosEquation,                  genNegEquation, isCorrect, writeJson
from neuralAlgonometry import catJsons

# path to trigonometry equations collected from wiki
inputPath = "axioms/trigonometryAxioms.json"
# path to some axioms from elementary algebra
axiomPath = "axioms/algebraAxioms.txt"
jsonAtts = ["equation", "range", "variables","labels"]

labels = []
inputRanges = []
inputEquations = [] 
eqVariables = []
parsedEquations = []
parsedRanges = []
ranges = []
inputAxioms = []
axiomVariables = []
axioms = []
axLabels = []

random.seed(1)

# parses latex equations from file:
readJson(inputPath, inputEquations, inputRanges, jsonAtts)
inputEquations[1]

# Converts latex equations to sympy equations using process_latex
parseEquation(inputEquations, parsedEquations, eqVariables)
parsedEquations[1]

# converts equations from sympy format to EquationTree object
# equations pretty print as well as pre order traversal follows
equations = []
for i, eq in enumerate(parsedEquations):
    # building EquationTree object
    currEq = buildEq(eq, eqVariables[i])
    # assigning a unique number to each node in the tree as well as indicating subtree depth at each level
    currEq.enumerize_queue()
    equations.append(currEq)
    
    # creating training labels
    # the first equation in the input function is incorrect. It has been deliberately added
    # to include all possible functionalities in the functionDictionary. 
    # This is for compatibility with MxNet's bucketingModule.
    if i == 0:
        labels.append(mx.nd.array([0]))
    else:
        labels.append(mx.nd.array([1]))
    
print "currEq:", equations[1]
print "pre order traversal"
equations[1].preOrder()

# parses text equations using the compiler package and returns an equation in the compiler format
readAxioms(axiomPath, inputAxioms, axiomVariables)
inputAxioms[0]

# converting compiler object axioms to EquationTree objects and creating training labels
for i, ax in enumerate(inputAxioms):
    currAxiom = buildAxiom(ax, axiomVariables[i])
    currAxiom.enumerize_queue()
    axioms.append(currAxiom)
    axLabels.append(mx.nd.array([1]))
    
print "an axiom:", axioms[0]
print "pre order traversal:"
axioms[0].preOrder()

# appending algebra axioms to trigonometry axioms
equations.extend(axioms)
eqVariables.extend(axiomVariables)
labels.extend(axLabels)
print len(equations)
print len(eqVariables)
print len(labels)

depthMat = [0 for _ in range(26)]
for eq in equations:
    depthMat[eq.depth] += 1
print "distribution of depth of equations"
print depthMat[:10]

# constructing the mathDictionary whose (key,value) pairs are valid math equalities
# e.g. (x+y : y+x) is a (key,value) pair in the mathDictionary
# the dictionary will be updated as more correct equations are generated
mathDictionary = {}
strMathDictionary = {}
for i, eq in enumerate(equations):
    if i!=0:
        eqCopy = copy.deepcopy(eq)
        if str(eqCopy) not in strMathDictionary:
            strMathDictionary[str(eqCopy)] = 1
            mathDictionary[eqCopy.args[0]] = eqCopy.args[1]
            mathDictionary[eqCopy.args[1]] = eqCopy.args[0]
        else:
            strMathDictionary[str(eqCopy)] += 1
# for k, v in strMathDictionary.iteritems():
#     print k, v
print len(strMathDictionary)
print len(mathDictionary)

maxDepth = 7
numPosEq = 10
numNegEq = 10
numNegRepeats = 2
thrsh = 5

# set maxDepthSoFar to 0 to generate up to thrsh number of 
# repeated equations before moving to equations of higher depth
maxDepthSoFar = 7
totDisc = 0
for i in range(0, numPosEq):
    randInd = random.choice(range(1, len(equations)))
    while labels[randInd].asnumpy() == 0:
        randInd = random.choice(range(1, len(equations)))
    randEq = copy.deepcopy(equations[randInd])
    randEqVariable = copy.deepcopy(eqVariables[randInd])

    posEq = genPosEquation(randEq, mathDictionary, randEqVariable)
    posVars = posEq.extractVars()
    posEq.enumerize_queue()

    old = 0
    disc = 0
    tries = 0
    # this loop is to make sure there are no repeats and also that enough 
    # number of equations of a certain depth are generated
    while str(posEq) in strMathDictionary or posEq.depth > maxDepthSoFar:
        if str(posEq) in strMathDictionary:
            strMathDictionary[str(posEq)] += 1
            old += 1
            totDisc += 1
        elif posEq.depth > maxDepthSoFar:
            disc += 1
            totDisc += 1

        if old > thrsh:
            old = 0
            maxDepthSoFar += 1
            print "new max depth %d" %(maxDepthSoFar)
            if maxDepthSoFar > maxDepth:
                print "reached maximum depth"
                maxDepthSoFar = maxDepth
                break

        randInd = random.choice(range(1, len(equations)))
        randEq = equations[randInd]
        randEqVariable = copy.deepcopy(eqVariables[randInd])
        posEq = genPosEquation(randEq, mathDictionary, randEqVariable)
        posVars = posEq.extractVars()
        posEq.enumerize_queue()

    if posEq.depth <= maxDepth:
        posEqCopy = copy.deepcopy(posEq)

        if str(posEqCopy) not in strMathDictionary:
            strMathDictionary[str(posEqCopy)] = 1
            if posEqCopy.args[0] not in mathDictionary:
                mathDictionary[posEqCopy.args[0]] = posEqCopy.args[1]
            if posEqCopy.args[1] not in mathDictionary:
                mathDictionary[posEqCopy.args[1]] = posEqCopy.args[0]

            equations.append(posEq)
            eqVariables.append(posVars)
            labels.append(mx.nd.array([1]))
    else:
        totDisc += 1
        print "discarded pos equation of depth greater than %d: %s" %(maxDepth, str(posEq))

depthMat = [0 for _ in range(26)]
for eq in equations:
    depthMat[eq.depth] += 1
print "distribution of depth of equations"
print depthMat

# generating negative equations
negLabels= [[] for _ in range(numNegRepeats)]
negEquations = [[] for _ in range(numNegRepeats)]
negEqVariables = [[] for _ in range(numNegRepeats)]
negStrMathDictionary = {}
corrNegs = 0
totDiscNeg = 0
ii = len(equations)
for i in range(1, len(equations)): 
    for rep in range(numNegRepeats):
        randInd = i
        randEq = copy.deepcopy(equations[i])
        randEqVariable = copy.deepcopy(eqVariables[randInd])

        negEq = genNegEquation(randEq, randEqVariable)
        negVars = negEq.extractVars()
        negEq.enumerize_queue()
        disc = 0
        tries = 0
        old = 0
        while str(negEq) in negStrMathDictionary or negEq.depth > maxDepth:
            if str(negEq) in negStrMathDictionary:
                negStrMathDictionary[str(negEq)] += 1
                old += 1
                totDiscNeg += 1
                # print "repeated neg equation"
            elif negEq > maxDepth:
                # print "equation larger than depth"
                disc += 1
                totDiscNeg += 1

            if old > thrsh:
                old = 0
                break

            negEq = genNegEquation(randEq, randEqVariable)
            negVars = negEq.extractVars()
            negEq.enumerize_queue()

        if negEq.depth <= maxDepth:
            
            negEqCopy = copy.deepcopy(negEq)
            try:
                isCorrect(negEq)

                if isCorrect(negEq):
                    corrNegs += 1

                    print "correct negative Eq:", negEq

                    if str(negEq) not in strMathDictionary:

                        strMathDictionary[str(negEqCopy)] = 1
                        if negEqCopy.args[0] not in mathDictionary:
                            mathDictionary[negEqCopy.args[0]] = negEqCopy.args[1]
                        if negEqCopy.args[1] not in mathDictionary:
                            mathDictionary[negEqCopy.args[1]] = negEqCopy.args[0]

                        labels.append(mx.nd.array([1]))
                        equations.append(negEq)
                        eqVariables.append(negVars)

                elif str(negEqCopy) not in negStrMathDictionary:
                        negStrMathDictionary[str(negEqCopy)] = 1

                        negLabels[rep].append(mx.nd.array([0]))
                        negEquations[rep].append(negEq)
                        negEqVariables[rep].append(negVars)
                else:
                    totDiscNeg += 1

            except:

                if str(negEqCopy) not in negStrMathDictionary:
                    negStrMathDictionary[str(negEqCopy)] = 1

                    negLabels[rep].append(mx.nd.array([0]))
                    negEquations[rep].append(negEq)
                    negEqVariables[rep].append(negVars)
                else:
                    totDiscNeg += 1

        else:
            totDiscNeg += 1
            print "discarded neg equation of depth greater than %d: %s" %(maxDepth, str(negEq))

depthMat = [0 for _ in range(26)]
for eq in negEquations[0]:
    depthMat[eq.depth] += 1
print "distribution of depth of neg equations"
print depthMat

# writing equations to file
writeJson("data/data%d_pos.json"%(numPosEq), equations, ranges, eqVariables, labels, maxDepth)
for rep in range(numNegRepeats):
    writeJson("data/data%d_neg_rep%d.json"%(numNegEq,rep), negEquations[rep], ranges, negEqVariables[rep], negLabels[rep], maxDepth)

catJsons(['data/data%d_pos.json'%(numPosEq), 'data/data%d_neg_rep%d.json'%(numNegEq,0), 'data/data%d_neg_rep%d.json'%(numNegEq,1)],
          'data/data%d_final.json'%(numPosEq), maxDepth=maxDepth)



