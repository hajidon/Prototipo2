import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

stopwordsNLTK = nltk.corpus.stopwords.words('portuguese')
stopwordsNLTK.append('vou')
stopwordsNLTK.append('tão')
stopwordsNLTK.append('não')

dataset = (pd.read_csv('Tweets_Mg.csv'))
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

def remove_stopwords(word_list):
    processed_word_list = []
    for word in word_list:
        word = word.lower()  # in case they arenet all lower cased
        semStop = [p for p in word.split() if p not in stopwordsNLTK]
        retorno = ' '.join(semStop)
        processed_word_list.append(retorno)
    return processed_word_list
def aplica_stemmer(word_list):
    stemmer = nltk.stem.RSLPStemmer()
    frasesStemming = []
    for word in word_list:
        comStemming = [str(stemmer.stem(p)) for p in word.split()]
        retorno = ' '.join(comStemming)
        frasesStemming.append((retorno))
    return frasesStemming

tweets_limpos = remove_hashtag(tweets)
tweets_limpos1 = remove_url(tweets_limpos)
tweets_sem_stop_word = remove_stopwords(tweets_limpos1)
tweets_com_steeming = aplica_stemmer(tweets_sem_stop_word)
vectorizer = TfidfVectorizer(encoding='utf-8', strip_accents='ascii', ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets_sem_stop_word)
#print(freq_tweets)
#modelo = LogisticRegression()
modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)

testes = ['Esse governo está no início, vamos ver o que vai dar',
         'esse governo é otimo',
         'O estado de Minas Gerais decretou calamidade financeira!!!',
         'A segurança desse país está deixando a desejar',
         'O governador de Minas é do PT',
         'Esse governo é o pior que ja vi',
         'Espero que seja um bom governo',
         'Estou muito triste com o governo',
         'eu estou gostando desse governo','Governo de Minas investiga casos suspeitos de febre amarela e malária no estado']

teste_Classe = ['neutro','positivo','negativo','negativo','neutro','negativo','neutro','negativo','positovo','negativo']
t1 = remove_url(testes)
t2 = remove_hashtag(t1)
t3 = remove_stopwords(t2)
t4 = aplica_stemmer(t3)
freq_testes = vectorizer.transform(t4)
teste = modelo.predict(freq_testes)
print(teste)
sentimento=['Positivo','Negativo','Neutro']
#confusion_matrix = confusion_matrix(teste_Classe,teste)
#print(confusion_matrix)
resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)
#resultados = modelo.score(freq_tweets,classes)
print(metrics.accuracy_score(classes,resultados))
print(metrics.classification_report(classes,resultados,sentimento),'')
print(pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')
#print(resultados)
#print(modelo.predict("Estou muito triste com o governo","negativo"))