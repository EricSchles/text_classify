import algorithms

def test_svm():
    testing = [("hello there","greeting"),("later","goodbye")]
    cl = algorithms.svm(testing)
    test = algorithms.preprocess("hello there friends")
    assert cl.classify(test) == "greeting"
    
def test_naive_bayes():
    testing = [("hello there","greeting"),("later","goodbye")]
    cl = algorithms.naive_bayes(testing)
    test = "Hello there friends"
    assert cl.classify(test) == "greeting"

def test_decision_tree():
    testing = [("hello there","greeting"),("later","goodbye")]
    cl = algorithms.decision_tree(testing)
    test = algorithms.preprocess("hello there friends")
    assert cl.classify(test) == "greeting"

def test_textrank():
    ranker = algorithms.textrank("hello there friends how are you")
    assert ranker.keyphrases
    assert ranker.summary
