from pandas import DataFrame


def createViterbiMatrix(inputSentence, wordPosProbabilityMap, posMarchovChainMap):
    """Creates the Viterbi decoding matrix for a given input sentence"""
    # Tokenize input sentence
    tokens = inputSentence.split()
    tokens.insert(0, "<start>")
    tokens.append("<stop>")

    # Get list of all possible POS tags
    poss = list(posMarchovChainMap.keys())

    # Initialize viterbi matrix
    viterbiMatrix = [[(0, None) for x in range(len(tokens))] for y in range(len(poss))]
    viterbiMatrix[0][0] = 1, None

    for tokenIndex in range(1, len(tokens)):
        token = tokens[tokenIndex]
        previousColumnIndex = tokenIndex - 1
        for posIndexFrom, posFrom in enumerate(poss):
            previousProbability, previousPointer = viterbiMatrix[posIndexFrom][previousColumnIndex]
            for posIndexTo, posTo in enumerate(poss):
                wordPosProbability = wordPosProbabilityMap.get(token).get(posTo, 0)
                posTransitionProbability = posMarchovChainMap.get(posFrom).get(posTo, 0)
                newPointer = (posIndexFrom, previousColumnIndex)
                newProbability = previousProbability * wordPosProbability * posTransitionProbability
                currentProbability, currentPoint = viterbiMatrix[posIndexTo][tokenIndex]
                if currentProbability < newProbability:
                    viterbiMatrix[posIndexTo][tokenIndex] = newProbability, newPointer

    viterbiDF = DataFrame(viterbiMatrix, index=poss, columns=tokens)
    return viterbiDF


def backtraceViterbiDF(vertbiDF):
    """Return the POS list found in the given vertibi matrix"""
    prediction = ["<stop>"]
    probability, pointer = vertbiDF["<stop>"]["<stop>"]
    while pointer:
        pos = vertbiDF.index[pointer[0]]
        prediction.insert(0, pos)
        probability, pointer = vertbiDF.iloc[pointer[0]].iloc[pointer[1]]
    return prediction


def main():
    # In class test case example to prove that the implementation works
    posMarchovChainMap = {
        "<start>": {
            "noun": 0.8,
            "verb": 0.2,
        },
        "noun": {
            "noun": 0.1,
            "verb": 0.8,
            "<stop>": 0.1,
        },
        "verb": {
            "noun": 0.2,
            "verb": 0.1,
            "<stop>": 0.7,
        },
        "<stop>": {
        }
    }
    wordPosProbabilityMap = {
        "fish": {
            "verb": 0.5,
            "noun": 0.8
        },
        "sleep": {
            "verb": 0.5,
            "noun": 0.2
        },
        "<start>": {
            "<start>": 1
        },
        "<stop>": {
            "<stop>": 1
        }
    }
    inputSentence = "fish sleep"

    viterbiDF = createViterbiMatrix(inputSentence, wordPosProbabilityMap, posMarchovChainMap)
    print("------In Class Test Case------")
    print("Input Sentence:", inputSentence)
    print("Viterbi Matrix:\n", viterbiDF)
    print("POS Output:", backtraceViterbiDF(viterbiDF))
    print()
    # The answer to the problem homework problem
    posMarchovChainMap = {
        "<start>": {
            "noun": 0.2,
            "verb": 0.3,
        },
        "noun": {
            "noun": 0.1,
            "verb": 0.3,
            "adverb": 0.1,
        },
        "verb": {
            "noun": 0.4,
            "verb": 0.1,
            "adverb": 0.4,
        },
        "adverb": {
            "<stop>": 0.1
        },
        "<stop>": {
        }
    }
    wordPosProbabilityMap = {
        "learning": {
            "verb": 0.003,
            "noun": 0.001
        },
        "throughly": {
            "adverb": 0.002
        },
        "changes": {
            "verb": 0.004,
            "noun": 0.003
        },
        "<start>": {
            "<start>": 1
        },
        "<stop>": {
            "<stop>": 1
        }
    }
    inputSentence = "learning changes throughly"

    viterbiDF = createViterbiMatrix(inputSentence, wordPosProbabilityMap, posMarchovChainMap)
    print("------Homework Answer------")
    print("Input Sentence:", inputSentence)
    print("Viterbi Matrix:\n", viterbiDF)
    print("POS Output:", backtraceViterbiDF(viterbiDF))
    print()


if __name__ == '__main__':
    main()
