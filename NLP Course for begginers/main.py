import nltk
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

def remove_pun(text):
    text_no_punc = "".join([word.lower() for word in text if word not in string.punctuation])
    return text_no_punc

def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

def remove_stopwords(stopwords, tokenized_text):
    clean_text = [word for word in tokenized_text if word not in stopwords]
    return clean_text

def stemming(tokenized_text):
    ps = nltk.PorterStemmer()
    text = [ps.stem(word) for word in tokenized_text]
    return text

#lemmatize is more accurate than stemming but it is slower
def lemmatize(tokenized_text):
    wn = nltk.WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

def clean_text(text):
    #remove punctuation
    text_no_punc = remove_pun(text)
    #tokenize
    tokenized_text = tokenize(text_no_punc)
    #tokenized text witout stopwords (ive should be removed)
    stopwords = nltk.corpus.stopwords.words('english')
    text_nostopwords = remove_stopwords(stopwords, tokenized_text)
    #lemmatize
    clean_text = lemmatize(text_nostopwords)

    return clean_text

def print_dataset_info(dataset):
    print(f'Input data has {len(dataset)} rows and {len(dataset.columns)} columns')
    print(f'Out of {len(dataset)} rows, {len(dataset[dataset["label"] == "spam"])} are spam and {len(dataset[dataset["label"] == "ham"])} are ham')
    print(f'Number of nulls in label: {dataset["label"].isnull().sum()}')
    print(f'Number of nulls in body_text: {dataset["body_text"].isnull().sum()}')

def test_regex():
    test = 'This is a made up string to test 2 different regex methods'
    test_messy = 'This     is a made up     string to test 2     different regex methods'
    test_messy1 = 'This-is-a-made/up.string*to>>>>test----2""""""diferent~regex.methods'

    print("test.split \s (single space)")
    print(re.split("\s", test))

    print("test_messy.split \s (single space)")
    print(re.split("\s", test_messy))

    print("test_messy.split \s+  (multiple spaces)")
    print(re.split("\s+", test_messy))

    print("test_messy1.split \s+ (multiple spaces)")
    print(re.split("\s+", test_messy1))

    print("test_messy1.split \W+ (everything except special characters)")
    print(re.split("\W+", test_messy1))

    print("test.findall \S+ (everything except spaces)")
    print(re.findall("\S+", test))

    print("test_messy.findall \S+ (everything except spaces)")
    print(re.findall("\S+", test_messy))

    print("test_messy1.findall \S+ (everything except spaces)")
    print(re.findall("\S+", test_messy1))

    print("test_messy1.findall \w+ (everything except special characters)")
    print(re.findall("\w+", test_messy1))


    pep8_test = 'I try to follow PEP8 guidelines'

    print("pep8_test.findall [a-z]+ (lowercase characters)")
    print(re.findall("[a-z]+", pep8_test))
    
    print("pep8_test.findall [A-Z]+ (uppercase characters)")
    print(re.findall("[A-Z]+", pep8_test))

    print("pep8_test.findall [A-Z]+[0-9]+ (uppercase characters followed by numbers)")
    print(re.findall("[A-Z]+[0-9]+", pep8_test))

    print("pep8_test.sub [A-Z]+[0-9]+ (uppercase characters followed by numbers)")
    print(re.sub("[A-Z]+[0-9]+", "PEP8 Python Styleguide", pep8_test))

    
 
def main():
    dataset = pd.read_csv('SMSSpamCollection.tsv', sep='\t', header=None)
    dataset.columns = ['label', 'body_text']
    
    #count vectorizer used to train the ML model
    count_vect = CountVectorizer(analyzer=clean_text)
    x_counts = count_vect.fit_transform(dataset['body_text'])
    #print(x_counts.shape)
    #print(count_vect.get_feature_names_out())
    
    #document matrix which stores the count of the words used on each file
    dataset_vect = pd.DataFrame(x_counts.toarray())

    #separate dependent and independet variables
    x = x_counts.toarray() #features
    y = dataset.iloc[:, 0].values #etiquetas de cada fila

    labelencoder = LabelEncoder()
    #tranformamos las etiquetas en numeros para su clasificacion
    y = labelencoder.fit_transform(y)

    #x_train y y_train van a ser los valores que vamos a utilizar para entrenar al modelo
    #x_test y y_test van a ser los valores utilizados para realizar las predicciones
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    #fitting naive bayes to the training set
    classifer = GaussianNB()
    classifer.fit(x_train, y_train)

    #predicting the test results and see if the result is the same as y_test
    y_pred = classifer.predict(x_test)

    #making the confussion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print(f'Model accuracy {(cm[0][0]+cm[1][1]) / (cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])}')


    



    

if __name__ == '__main__':
    main()