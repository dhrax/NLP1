from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps= PorterStemmer()
words= ["gaming","games","game"]
for w in words:
    print(w,":", ps.stem(w))
