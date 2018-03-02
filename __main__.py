from typing import List, Dict
import pandas
import pprint
from pandas import DataFrame
import numpy


def createViterbiMatrix(inputSentence: str,
                        wordPosProbabilityMap,
                        posMarchovChainMap) -> DataFrame:
    columns: List[str] = inputSentence.split()
    columns.insert(0, "<start>")
    columns.append("<stop>")
    rows: List[str] = list(posMarchovChainMap.keys())

    viterbiMatrix = [[(0, None) for x in range(len(rows))] for y in range(len(columns))]
    pp = pprint.PrettyPrinter(depth=6, width=300)
    viterbiMatrix[0][0] = 1, None
    # pp.pprint(viterbiMatrix)

    for columnIndex in range(1, len(columns)):
        token = columns[columnIndex]
        previousColumnIndex = columnIndex - 1
        for rowIndexFrom, posFrom in enumerate(rows):
            previousProbability, previousPoint = viterbiMatrix[rowIndexFrom][previousColumnIndex]
            for rowIndexTo, posTo in enumerate(rows):
                wordPosProbability = wordPosProbabilityMap.get(token).get(posTo, 0)
                stateChangeProbability = posMarchovChainMap.get(posFrom).get(posTo, 0)
                newPointer = (rowIndexFrom, previousColumnIndex)
                newScore = previousProbability * wordPosProbability * stateChangeProbability
                currentScore, currentPoint = viterbiMatrix[rowIndexTo][columnIndex]
                if currentScore < newScore:
                    viterbiMatrix[rowIndexTo][columnIndex] = newScore, newPointer
    # pp.pprint(viterbiMatrix)
    viterbiDF = DataFrame(viterbiMatrix, index=rows, columns=columns)
    return viterbiDF


def backtraceViterbiDF(vertbiDF):
    prediction = ["<end>"]
    probability, pointer = vertbiDF["<stop>"]["<stop>"]
    while pointer:
        pos = vertbiDF.index[pointer[0]]
        prediction.insert(0, pos)
        probability, pointer = vertbiDF.iloc[pointer[0]].iloc[pointer[1]]
    return prediction


def main():
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
    print(inputSentence)
    print(viterbiDF)
    print(backtraceViterbiDF(viterbiDF))

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
    print(inputSentence)
    print(viterbiDF)
    print(backtraceViterbiDF(viterbiDF))


if __name__ == '__main__':
    main()
