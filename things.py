from text_classify import algorithms
testing = [
    ("Hello","greeting"),
    ("Hi","greeting"),
    ("Hello there","greeting"),
    ("How are you?","greeting"),
    ("Wazzup?"),("greeting"),
    ("Hey!","greeting"),
    ("hey.","greeting"),
    ("hi.","greeting"),
    ("Hi there","greeting"),
    ("Heyy","greeting"),
    ("Hello, how are you?","greeting"),
    ("bye","goodbye"),
    ("goodbye","goodbye"),
    ("byee","goodbye"),
    ("later","goodbye"),
    ("bye bye","goodbye"),
    ("adios","goodbye"),
    ("ciao","goodbye"),
    ("see ya","goodbye")
]
cl = algorithms.svm(testing)
print cl.classify(algorithms.preprocess("byee"))
algorithms.cross_validate(testing,model="svm")
