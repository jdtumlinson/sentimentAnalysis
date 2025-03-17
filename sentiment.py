# CS331 Sentiment Analysis Assignment 3
# This file contains the processing functions

import string
import csv
import classifier



def process_text(text):
    """
    Preprocesses the text: Remove apostrophes, punctuation marks, etc.
    Returns a list of text
    """
    processed_text = []
    
    for line in text:
        line = line.lower()  # Convert to lowercase
        line = line.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        processed_text += [line]
        
    return processed_text



def build_vocab(text):
    """
    Builds the vocab from the preprocessed text
    preprocessed_text: output from process_text
    Returns unique text tokens
    """

    vocab = set()

    for line in text:
        words = line.split()[:-1]   # Don't include labels
        vocab.update(words)  # add words to vocab set

    vocab = sorted(vocab) # sort alphabetically

    return vocab



def vectorize_text(text, vocab):
    """
    Converts the text into vectors
    text: preprocess_text from process_text
    vocab: vocab from build_vocab
    Returns the vectorized text and the labels
    """

    features = []
    labels = []

    for idx, line in enumerate(text):
        words = line.split()[:-1]       #Don't include labels

        vector = [1 if word in words else 0 for word in vocab]

        # label handling

        label = int(text[idx].strip()[-1]) # grabs label at end
        labels.append(label) # ! error here? being added twice?

        features.append(vector)

    return features, labels



def accuracy(predicted_labels, true_labels):
    """
    predicted_labels: list of 0/1s predicted by classifier
    true_labels: list of 0/1s from text file
    return the accuracy of the predictions
    """

    # (# correctly predicted class labels) / (total # of predictions)
    totalCorrct = sum(1 if predicted_labels[idx] == true_labels[idx] else 0 for idx in range(len(true_labels))) 

    return totalCorrct / len(true_labels)



def write_vectors(out, vocab, features):
    with open(out, "w", newline="", encoding="utf-8") as file:
        csv_write = csv.writer(file)
        csv_write.writerow(vocab + ["classlabel"])
        csv_write.writerows(features)




def extract_text(filename):
    lineList = []
    
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            lineList += [line]
    
    return lineList



def main():
    # Take in text files and outputs sentiment scores

    train_text = extract_text("trainingSet.txt")
    test_text = extract_text("testSet.txt")
    
    train_text = process_text(train_text)
    test_text = process_text(test_text)
    
    chunk = len(train_text) // 4
    chunkTotal = 0
    
    with open("results.txt", "w") as file: file.write("")

    for idx, section in enumerate([train_text[: chunk], train_text[: chunk * 2], train_text[: chunk * 3], train_text]):
        vocab = build_vocab(section)
        testVocab = build_vocab(test_text)
        
        chunkTotal += chunk
        
        train_vectors, train_labels = vectorize_text(section, vocab)
        test_vectors, test_labels = vectorize_text(test_text, testVocab)
        
        write_vectors("preprocessed_train.txt", vocab, train_vectors)
        write_vectors("preprocessed_test.txt", vocab, test_vectors)
        
        bayesClassifier = classifier.BayesClassifier()
        bayesClassifier.train(train_vectors, train_labels, vocab)
        
        train_preidctions = bayesClassifier.classify_text(train_vectors, vocab)
        test_predictions = bayesClassifier.classify_text(test_vectors, testVocab)
        
        testAccuracyPercentage = accuracy(test_predictions, test_labels)
        trainAccuracyPercentage = accuracy(train_preidctions, train_labels)
        
        print(f"({idx + 1})Training Set Accuracy: {round(trainAccuracyPercentage, 6) * 100}%")
        print(f"({idx + 1})Testing Set Accuracy: {round(testAccuracyPercentage, 6) * 100}%")
        
        text = f"({idx + 1}) Training Set Accuracy: {round(trainAccuracyPercentage, 6) * 100}%     |     Examples Used: {chunkTotal}\n({idx + 1}) Testing Set Accuracy: {round(testAccuracyPercentage, 6) * 100}%\n\n"
        
        with open("results.txt", "a", newline="", encoding="utf-8") as file:
            file.write(text)
    
    return 1


if __name__ == "__main__":
    main()