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
from sklearn import cross_validation
import distance

def textrank(text):
    return TextRank(text=text)

def TfIdf(document_list):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(document_list)
    return dict(zip(vectorizer.get_feature_names(),vectorizer.idf_))

def cosine_similarity(documentA,documentB):
    docs = [documentA,documentB]
    tfidf = TfIdf(docs)
    cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten() 
    return cosine_similarities


def str_comp(str1,str2):
    score = 0
    words1 = str1.split(" ")
    words2 = str2.split(" ")
    if len(words1) < len(words2):
        for ind,word in enumerate(words1):
            score += word_comp(word,words2[ind])
    else:
        for ind,word in enumerate(words2):
            score += word_comp(word,words1[ind])
    return score

def word_comp(str1,str2):
    subword1 = [str1[:pos] for pos in xrange(1,len(str1)+1)]
    subword2 = [str2[:pos] for pos in xrange(1,len(str2)+1)]
    score = 0
    if len(subword1) < len(subword2):
        for ind,sub in enumerate(subword1):
            if sub == subword2[ind]:
                score += 1
        return score/float(len(subword1)*2)
    else:
        for ind,sub in enumerate(subword2):
            if sub == subword1[ind]:
                score += 1
        return score/float(len(subword2)*2)


def ngram(sentence,n):
    input_list = [elem for elem in sentence.split(" ") if elem != '']
    return zip(*[input_list[i:] for i in xrange(n)])

def accuracy(classifier_name,classifier,test_data):
    """
    example usage: 
    
    from text_classify.algorithms import *
    import pickle
    training_data = pickle.load(open("training_data","rb"))
    test_data = pickle.load(open("test_data,"rb"))
    cl = svm(training_data)
    accuracy("svm",cl,test_data)
    
    the possible choices for classifiers at present are:
       svm, decision_tree, naive_bayes
    (more to come soon!)
    """
    if classifier_name in ["svm","decision_tree"]:
        testing = [preprocess(data[0],label=data[1]) for data in test_data]
        counter = 0.0
        for ind,data in enumerate(test_data):
            if classifier.classify(testing[ind][0]) == data[1]:
                counter += 1
        return counter/len(test_data)
    if classifier_name == "naive_bayes":
        return classifier.accuracy(test_data)
    else:
        return "no idea!"
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
def svm(train_data,preprocessing=True):
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

def cross_val(data,model=None):
    training_set = nltk.classify.apply_features(preprocess,data)
    cv = cross_validation.KFold(len(training_set), n_folds=10, indices=True, shuffle=False, random_state=None)
    if model == "svm" or model=="SVM":
        svm = SklearnClassifier(LinearSVC())

        for traincv, testcv in cv:
            classifier = svm.train(training_set[traincv[0]:traincv[len(traincv)-1]])
            print 'accuracy:', nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])
