"""
citations:
- TA's discussion slides
- ECS189G textbook 
- TA helped me during Office Hours
- chinese blog that explains the concept and equation (url:http://www.shuang0420.com/2017/06/01/NLP%20%E7%AC%94%E8%AE%B0%20-%20Sentiment%20Analysis/)
- piazza's replies
"""

import sys
import getopt
import os
import math
import operator
import collections
class NaiveBayes:
    class TrainSplit:
        """
        Set of training and testing data
        """
        def __init__(self):
            self.train = []
            self.test = []

    class Document:
        """
        This class represents a document with a label. classifier is 'pos' or 'neg' while words is a list of strings.
        """
        def __init__(self):
            self.classifier = ''
            self.words = []

    def __init__(self):
        """
        Initialization of naive bayes
        """
        self.stopList = set(self.readFile('data/english.stop'))
        self.bestModel = False
        self.stopWordsFilter = False
        self.naiveBayesBool = False
        self.numFolds = 10
        self.numDocs = 0
        self.posWord = collections.defaultdict(lambda: 0) # posWord["Word"] returns Count(Word, Positive)
        self.negWord = collections.defaultdict(lambda: 0) # negWord["Word"] returns Count(Word, Negative)
        self.posDocCount = collections.defaultdict(lambda: 0)
        self.negDocCount = collections.defaultdict(lambda: 0)
        self.numPos = 0
        self.numNeg = 0
        self.numPosDoc = 0
        self.numNegDoc = 0
        self.count = 0
        self.vocabSize = set()
        # TODO
        # Implement a multinomial naive bayes classifier and a naive bayes classifier with boolean features. The flag
        # naiveBayesBool is used to signal to your methods that boolean naive bayes should be used instead of the usual
        # algorithm that is driven on feature counts. Remember the boolean naive bayes relies on the presence and
        # absence of features instead of feature counts.

        # When the best model flag is true, use your new features and or heuristics that are best performing on the
        # training and test set.

        # If any one of the flags filter stop words, boolean naive bayes and best model flags are high, the other two
        # should be off. If you want to include stop word removal or binarization in your best performing model, you
        # will need to write the code accordingly.
        #0.779, 0.824
    def classify(self, words):
        """
        Classify a list of words and return a positive or negative sentiment
        """
        if self.stopWordsFilter:
            words = self.filterStopWords(words)

        # TODO
        # classify a list of words and return the 'pos' or 'neg' classification
        # Write code here
        posScore = 0.0
        negScore = 0.0

        """Baseline model"""
        if not self.naiveBayesBool and not self.bestModel:
    
            posScore += math.log(self.numPosDoc) - math.log(self.numDocs)
            negScore += math.log(self.numNegDoc) - math.log(self.numDocs)
            sumpos = sum(self.posWord.values())
            sumneg = sum(self.negWord.values())
            for word in words:
                # calculate positive sentiment
                posScore += math.log(self.posWord[word] + 1)
                posScore -= math.log(sumpos + 1 * self.count)
                # calculate negative sentiment
                negScore += math.log(self.negWord[word] + 1)
                negScore -= math.log(sumneg + 1 * self.count)

        """Binary model"""
        if self.naiveBayesBool and not self.bestModel:

            posScore += math.log(self.numPosDoc) - math.log(self.numDocs)
            negScore += math.log(self.numNegDoc) - math.log(self.numDocs)
            sumpos = sum(self.posWord.values())
            sumneg = sum(self.negWord.values())

            for word in words:
                 # calculate positive sentiment
                posScore += math.log(self.posDocCount[word] + 5)
                posScore -= math.log(sumpos + 5 * self.count)
                # calculate negative sentiment
                negScore += math.log(self.negDocCount[word] + 5)
                negScore -= math.log(sumneg + 5 * self.count)

        """Custom model"""
        if self.bestModel:
         
            posScore += math.log(self.numPosDoc) - math.log(self.numDocs)
            negScore += math.log(self.numNegDoc) - math.log(self.numDocs)
            sumpos = sum(self.posWord.values())
            sumneg = sum(self.negWord.values())
            alpha = 20

            for word in words:
                 # calculate positive sentiment
                posScore += math.log(self.posWord[word] + alpha)
                posScore -= math.log(sumpos + alpha * len(self.vocabSize))
                # calculate negative sentiment
                negScore += math.log(self.negWord[word] + alpha)
                negScore -= math.log(sumneg + alpha* len(self.vocabSize))


        if posScore > negScore:
            return 'pos'

        return 'neg'

    def addDocument(self, classifier, words):
        """
        Train your model on a document with label classifier (pos or neg) and words (list of strings). You should
        store any structures for your classifier in the naive bayes class. This function will return nothing
        """
        # TODO
        # Train model on document with label classifiers and words
        # Write code here

        if self.bestModel:
            self.numDocs += 1
            if classifier == 'pos':
                self.numPosDoc += 1
            elif classifier == 'neg':
                self.numNegDoc += 1

   
            if (classifier == 'pos'):
                for word in set(words):
                    self.posWord[word] += 1
                    self.vocabSize.add(word)
            else:
                for word in set(words):
                    self.negWord[word] += 1
                    self.vocabSize.add(word)


        else:
            self.numDocs += 1
            appearedBeforePositive = collections.defaultdict(int) # 0 for false, 1 for true
            appearedBeforeNegative = collections.defaultdict(int) # 0 for false, 1 for true
            

            if classifier == 'pos':
                self.numPosDoc += 1
            elif classifier == 'neg':
                self.numNegDoc += 1
            

            for word in words:


                if classifier == 'pos':
            
                    self.posWord[word] += 1
                    self.numPos += 1
                    if appearedBeforePositive[word] == 0:
                        appearedBeforePositive[word] = 1
                        self.posDocCount[word] += 1
                        self.count += 1

                elif classifier == 'neg':
                   
                    self.negWord[word] += 1
                    self.numNeg += 1
                    if appearedBeforeNegative[word] == 0:
                        appearedBeforeNegative[word] = 1
                        self.negDocCount[word] += 1
                        self.count += 1

    


    def readFile(self, fileName):
        """
        Reads a file and segments.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        str = '\n'.join(contents)
        result = str.split()
        return result

    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        for fileName in posDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            doc.classifier = 'pos'
            split.train.append(doc)
        for fileName in negDocTrain:
            doc = self.Document()
            doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            doc.classifier = 'neg'
            split.train.append(doc)
        return split

    def train(self, split):
        for doc in split.train:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)
            self.addDocument(doc.classifier, words)

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posDocTrain = os.listdir('%s/pos/' % trainDir)
        negDocTrain = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(doc)
                else:
                    split.train.append(doc)
            yield split

    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for doc in split.test:
            words = doc.words
            if self.stopWordsFilter:
                words = self.filterStopWords(words)
            guess = self.classify(words)
            labels.append(guess)
        return labels

    def buildSplits(self, args):
        """
        Construct the training/test split
        """
        splits = []
        trainDir = args[0]
        if len(args) == 1:
            print '[INFO]\tOn %d-fold of CV with \t%s' % (self.numFolds, trainDir)

            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fold in range(0, self.numFolds):
                split = self.TrainSplit()
                for fileName in posDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                    doc.classifier = 'pos'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                for fileName in negDocTrain:
                    doc = self.Document()
                    doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                    doc.classifier = 'neg'
                    if fileName[2] == str(fold):
                        split.test.append(doc)
                    else:
                        split.train.append(doc)
                splits.append(split)
        elif len(args) == 2:
            split = self.TrainSplit()
            testDir = args[1]
            print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
            posDocTrain = os.listdir('%s/pos/' % trainDir)
            negDocTrain = os.listdir('%s/neg/' % trainDir)
            for fileName in posDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                doc.classifier = 'pos'
                split.train.append(doc)
            for fileName in negDocTrain:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                doc.classifier = 'neg'
                split.train.append(doc)

            posDocTest = os.listdir('%s/pos/' % testDir)
            negDocTest = os.listdir('%s/neg/' % testDir)
            for fileName in posDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/pos/%s' % (testDir, fileName))
                doc.classifier = 'pos'
                split.test.append(doc)
            for fileName in negDocTest:
                doc = self.Document()
                doc.words = self.readFile('%s/neg/%s' % (testDir, fileName))
                doc.classifier = 'neg'
                split.test.append(doc)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """
        Stop word filter
        """
        removed = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                removed.append(word)
        return removed


def test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel):
    nb = NaiveBayes()
    splits = nb.buildSplits(args)
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = NaiveBayes()
        classifier.stopWordsFilter = stopWordsFilter
        classifier.naiveBayesBool = naiveBayesBool
        classifier.bestModel = bestModel
        accuracy = 0.0
        for doc in split.train:
            words = doc.words
            classifier.addDocument(doc.classifier, words)

        for doc in split.test:
            words = doc.words
            guess = classifier.classify(words)
            if doc.classifier == guess:
                accuracy += 1.0

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyFile(stopWordsFilter, naiveBayesBool, bestModel, trainDir, testFilePath):
    classifier = NaiveBayes()
    classifier.stopWordsFilter = stopWordsFilter
    classifier.naiveBayesBool = naiveBayesBool
    classifier.bestModel = bestModel
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testFile = classifier.readFile(testFilePath)
    print classifier.classify(testFile)


def main():
    stopWordsFilter = False
    naiveBayesBool = False
    bestModel = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    if ('-f', '') in options:
        stopWordsFilter = True
    elif ('-b', '') in options:
        naiveBayesBool = True
    elif ('-m', '') in options:
        bestModel = True

    if len(args) == 2 and os.path.isfile(args[1]):
        classifyFile(stopWordsFilter, naiveBayesBool, bestModel, args[0], args[1])
    else:
        test10Fold(args, stopWordsFilter, naiveBayesBool, bestModel)


if __name__ == "__main__":
    main()
