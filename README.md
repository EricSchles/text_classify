# text_classify
A general set of tools for text classification, ranking, feature extraction, and prediction

##Introduction/Intention

The goal of this tool is to make it easier to classify documents by providing a simple high level interface for a number of existing tools as well as be a place for novel algorithms to find use among users.

##Dependencies

You can also sudo pip install for each of these packages - I am working on a requirements.txt file
[install nltk](http://www.nltk.org/install.html)
[install textblob](http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/)
[install network x](https://networkx.github.io/download.html)
[install sci-kit learn](http://scikit-learn.org/stable/install.html)

download the nltk corpora:

```
import nltk
nltk.download()
```

##Installation

To install simply do the following:

`sudo python setup.py install`

This will install the package.  

##Some simple examples

###Naive Bayes Classification

```
from text_classify.algorithms import naive_bayes  
#Data appears as [([data to classify],[label]),..]
testing = [("hello there","greeting"),("later","goodbye")]
cl = naive_bayes(testing)
test = "Hello there friends"
cl.classify(test) # prints "greeting"
```

###Support Vector Machines

```
from text_classify.algorithms import svm, preprocess
#Data appears as [([data to classify],[label]),..]
testing = [("hello there","greeting"),("later","goodbye")]
cl = svm(testing)
test = preprocess("Hello there friends")
cl.classify(test) # prints "greeting"
```

###Decision Tree
```
from text_classify.algorithms import decision_tree, preprocess
#Data appears as [([data to classify],[label]),..]
testing = [("hello there","greeting"),("later","goodbye")]
cl = decision_tree(testing)
test = preprocess("Hello there friends")
cl.classify(test) # prints "greeting"
```

###Text Rank

```
ranker = algorithms.textrank("hello there friends how are you")
print ranker.keyphrases
print ranker.summary
```

##Current algorithms supported

* [TFIDF](http://www.tfidf.com/)
* [Cosine Simiarlity](https://en.wikipedia.org/wiki/Cosine_similarity)
* [SVM text classification](http://www.nltk.org/api/nltk.classify.html)
* [naive bayesian classification](http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/)
* [Text Rank](http://acl.ldc.upenn.edu/acl2004/emnlp/pdf/Mihalcea.pdf)
* [Expectation Maximization-algorithm](http://crow.ee.washington.edu/people/bulyko/papers/em.pdf)
* [N-grams](https://en.wikipedia.org/wiki/N-gram)

###ToDOs

* implement Deep Belief Networks
* implement neural networks
* create a high level interface to send jobs to spark and hadoop
