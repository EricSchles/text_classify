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
