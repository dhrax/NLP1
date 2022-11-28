from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

train = [('I like coffee.', 'pos'), ('I am playing basketball', 'pos'), ('I cannot read the whole book at once', 'neg'), ('Today is a bad day', 'neg')]


cl = NaiveBayesClassifier(train)
cl.classify('I am happy''pos')
blob = TextBlob("This movie is bad", classifier=cl)

for s in blob.sentences:
    print(s)
    print(s.classify())