from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob.classifiers import DecisionTreeClassifier as DTC
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier

def TfIdf(document_list):
    return TfidfVectorizer().fit_transform(document_list)

def cosine_similarity(documentA,documentB):
    docs = [documentA,documentB]
    tfidf = TfIdf(docs)
    cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten() 
    return cosine_similarities

# training data throughout: 
# expects a list of sets of the form:
# [("first words","first label"),("second words","second label"),..]
def naive_bayes(train_data):
    """
    cl.classify("some new text") #a label returned
    """
    return NBC(train_data)
#how to train: http://glowingpython.blogspot.com/2013/07/combining-scikit-learn-and-ntlk.html
def svm(train_data):
    cl = SklearnClassifier(LinearSVC())
    cl.train(train_data)
    return cl
