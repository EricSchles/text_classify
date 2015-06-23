from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob.classifiers import DecisionTreeClassifier as DTC
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
from nltk.classify import DecisionTreeClassifier
from nltk.classify import MaxentClassifier
import nltk
from textrank import TextRank

def textrank(text):
    return TextRank(text=text)

def TfIdf(document_list):
    return TfidfVectorizer().fit_transform(document_list)

def cosine_similarity(documentA,documentB):
    docs = [documentA,documentB]
    tfidf = TfIdf(docs)
    cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten() 
    return cosine_similarities

def ngram(sentence,n):
    input_list = [elem for elem in sentence.split(" ") if elem != '']
    return zip(*[input_list[i:] for i in xrange(n)])

# training data throughout: 
# expects a list of sets of the form:
# [("first words","first label"),("second words","second label"),..]
def naive_bayes(train_data):
    """
    cl.classify("some new text") #a label returned
    """
    return NBC(train_data)

#featurize words in a sentence for svm
#feature choice from this paper: http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
def preprocess(sentence,label=None):
    tokens = nltk.word_tokenize(sentence)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    features = {}
    for token in tokens:
        features[token]=tokens.count(token)
    if label:
        return (features,label)
    else:
        return features

#how to train: http://glowingpython.blogspot.com/2013/07/combining-scikit-learn-and-ntlk.html
#expectation is train_data is a tuple
def svm(train_data):
    training_data = []
    for data in train_data:
        training_data.append(preprocess(data[0],label=data[1]))
    cl = SklearnClassifier(LinearSVC())
    cl.train(training_data)
    return cl

def decision_tree(train_data):
    training_data = []
    for data in train_data:
        training_data.append(preprocess(data[0],label=data[1]))
    cl = DecisionTreeClassifier.train(training_data)
    return cl

