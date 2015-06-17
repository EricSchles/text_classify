import nltk
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
# def dialogue_act_features(sentence):
#     """
#         Extracts a set of features from a message.
#     """
#     features = {}
#     tokens = nltk.word_tokenize(sentence)
#     for t in tokens:
#         features['contains(%s)' % t.lower()] = True    
#     return features

# # data structure representing the XML annotation for each post
# posts = nltk.corpus.nps_chat.xml_posts() 
# # label set
# cls_set = ['Emotion', 'ynQuestion', 'yAnswer', 'Continuer',
# 'whQuestion', 'System', 'Accept', 'Clarify', 'Emphasis', 
# 'nAnswer', 'Greet', 'Statement', 'Reject', 'Bye', 'Other']
# featuresets = [] # list of tuples of the form (post, features)
# for post in posts: # applying the feature extractor to each post
#  # post.get('class') is the label of the current post
#  featuresets.append((dialogue_act_features(post.text),cls_set.index(post.get('class'))))

#  print featuresets[0]

def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    features = {}
    for token in tokens:
        features[token]=tokens.count(token)
    return features
featureset = []
sentences = [
    "hello there, how are you?  Are you very happy??",
    "Yammering on all the time, what a loser"
    ]
for sentence in sentences:
    features = preprocess(sentence)
    featureset.append(features)

cls = SklearnClassifier(LinearSVC())
featuresets = []
featuresets.append((featureset[0],"first"))
featuresets.append((featureset[1],"second"))
cls.train(featuresets)
print cls.classify(preprocess("hello there, friends"))
