# This file implements a Naive Bayes Classifier

import numpy


class BayesClassifier():
    """
    Naive Bayes Classifier
    file length: file length of training file
    sections: sections for incremental training
    """
    def __init__(self):
        self.postive_word_counts = {}
        self.negative_word_counts = {}
        self.percent_positive_scentences = 0
        self.percent_negative_scentences = 0
        self.file_length = 499
        self.file_sections = [self.file_length // 4, self.file_length // 3, self.file_length // 2]


    def train(self, train_data, train_labels, vocab):
        """
        This function builds the word counts and sentence percentages used for classify_text
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab
        """
        
        trainTable = {word: [0, 0] for word in vocab}  # {word: [count_in_pos, count_in_neg]}
        pPos, pNeg = 0, 0  # Total positive and negative sentences

        # Count positive and negative sentences
        for sentiment in train_labels:
            if sentiment == 1:
                pPos += 1
            else:
                pNeg += 1

        self.percent_positive_scentences = pPos
        self.percent_negative_scentences = pNeg

        # Count word occurrences
        for idx, sentence in enumerate(train_data):
            for word_idx, present in enumerate(sentence):
                if present == 1:  # Word appears in sentence
                    word = vocab[word_idx]
                    if train_labels[idx] == 1:
                        trainTable[word][0] += 1  # Increment positive count
                    else:
                        trainTable[word][1] += 1  # Increment negative count


        vocab_size = len(vocab)
        self.postive_word_counts = {
            word: (counts[0] + 1) / (pPos + vocab_size) for word, counts in trainTable.items()
        }
        self.negative_word_counts = {
            word: (counts[1] + 1) / (pNeg + vocab_size) for word, counts in trainTable.items()
        }
        
        

    def classify_text(self, vectors, vocab):
        """
        vectors: [vector1, vector2, ...]
        predictions: [0, 1, ...]
        """
        predictions = []
        vocab_size = len(vocab)

        # Prior probabilities
        prior_pos = numpy.log(self.percent_positive_scentences / (self.percent_positive_scentences + self.percent_negative_scentences))
        prior_neg = numpy.log(self.percent_negative_scentences / (self.percent_positive_scentences + self.percent_negative_scentences))

        for sentence in vectors:
            pos = prior_pos
            neg = prior_neg

            for word_idx, present in enumerate(sentence):
                if present == 0: continue
                word = vocab[word_idx]

                # Get probability of the word appearing in positive and negative classes
                pos += numpy.log(self.postive_word_counts.get(word, 1 / (self.percent_positive_scentences + vocab_size)))
                neg += numpy.log(self.negative_word_counts.get(word, 1 / (self.percent_positive_scentences + vocab_size)))

            predictions.append(1 if pos > neg else 0)
                
        return predictions