from text_classify import algorithms
testing = [("hello there","Phil"),("later","Gena")]
cl = algorithms.svm(testing)
test = algorithms.preprocess("hello there friends")
print cl.classify(test) == "Phil"
