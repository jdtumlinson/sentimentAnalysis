# Sentiment Analysis Assignment
This repository contains the code for implementing naive Bayes classification to determine the sentiment of a writers attitude towards the topics of which they are writing about. In the case of this repo, that topic is restruants. A large training and testing set of reviews have been gathered, each containing a given label -- 1 for positive and 0 for negative.
The training and testings sets are arranged into vectors containing the words usedin the sentence and the sentiment. Probabilities are then calculated based on the training data. When run through the `classify_text` method with a given vector of sentences, the program will use naive Bayes classification to tetermine the sentiment of the sentence.

# To Run
In order to run the program, enter `python3 sentiment.py`. This will output results to the terminal and into the `results.txt` file
