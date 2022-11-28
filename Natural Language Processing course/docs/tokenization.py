import nltk
nltk.download('punkt')
sentence = """Today I went for shopping"""
tokens = nltk.word_tokenize(sentence)
tags = nltk.pos_tag(tokens)

# print tokens
print(tokens)
print(tags)