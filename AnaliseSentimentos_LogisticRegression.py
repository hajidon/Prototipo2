import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import sklearn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

stopwordsNLTK = nltk.corpus.stopwords.words('portuguese')
stopwordsNLTK.append('vou')
stopwordsNLTK.append('tão')
stopwordsNLTK.append('não')

dataset = (pd.read_csv('Tweets_Mg1.csv'))
dataset.count()

tweets = dataset['Text'].values
classes = dataset['Classificacao'].values

def remove_hashtag(word_list):
    processed_word_list = []

    for word in word_list:
        limpo = " ".join(word.strip() for word in re.split('#|_|@', word))
        processed_word_list.append(limpo)

    return processed_word_list
def remove_url(word_list):
    processed_word_list = []

    for word in word_list:
        limpo = re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE)
        processed_word_list.append(limpo)

    return processed_word_list
#Diminui as palvras ao seu radical
def aplica_stemmer(word_list):
        stemmer = nltk.stem.RSLPStemmer()
        processed_word_list = []
        for word in word_list:
                comStemming = [str(stemmer.stem(p)) for p in word.split()]
                retorno = ' '.join(comStemming)
                processed_word_list.append(retorno)
        return processed_word_list

def remove_stopwords(word_list):
    processed_word_list = []
    for word in word_list:
        word = word.lower()  # in case they arenet all lower cased
        semStop = [p for p in word.split() if p not in stopwordsNLTK]
        retorno = ' '.join(semStop)
        processed_word_list.append(retorno)
    return processed_word_list

tweets_limpos = remove_hashtag(tweets)
tweets_limpos1 = remove_url(tweets_limpos)
tweets_sem_stop_word = remove_stopwords(tweets)
tweets_com_stemmer = aplica_stemmer(tweets)
#analyzer="word"
#ngram_range=(1,2)
vectorizer = TfidfVectorizer(encoding='utf-8', strip_accents='ascii', analyzer='word')
freq_tweets = vectorizer.fit_transform(tweets_com_stemmer)
#print(freq_tweets)
modelo = LogisticRegression()
#modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)

testes = ['Esse governo está no início, vamos ver o que vai dar',
         'Estou muito feliz com o governo de Minas esse ano',
         'O estado de Minas Gerais decretou calamidade financeira!!!',
         'A segurança desse país está deixando a desejar',
         'O governador de Minas é do PT',
         'Esse governo é o pior que ja vi',
         'Espero que seja um bom governo',
         'Estou muito triste com o governo',
         'eu estou gostando desse governo',
         'bandido bom é bandido morto']

t1 = remove_stopwords(testes)
freq_testes = vectorizer.transform(t1)
teste = modelo.predict(freq_testes)
print(teste)


resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)
print(metrics.accuracy_score(classes,resultados))
sentimento=['Positivo','Negativo']
print(metrics.classification_report(classes,resultados,sentimento),'')
print(pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')