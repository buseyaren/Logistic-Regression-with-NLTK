import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#eğitim boyutu
TRAIN_SIZE = 0.75
# parametreler
FILTER_STEM = True
TRAIN_PORTION = 0.8
RANDOM_STATE = 7

#veri setindeki verilerin çekilmesi
df = pd.read_csv('dataset/sentiment140.csv',
                 encoding="ISO-8859-1",
                 names=["target", "ids", "date", "flag", "user", "text"])
#Kategoriler
decode_map = {0: "NEGATİF", 2: "NÖTR", 4: "POZİTİF"}

df.target = df.target.apply(lambda x: decode_map[x])

print(df.target.value_counts())
#Preprocessing  -> Stopwords + Stemming
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

def filter_stopwords(text):
#durdurma sözcüklerinin ayrıştırılması, Preprocessing Aşaması
    text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(text).lower()).strip()

    if FILTER_STEM:   #TRUE
        return " ".join([stemmer.stem(token) for token in text.split() if token not in stop_words])
    else:             #FALSE
        return " ".join([token for token in text.split() if token not in stop_words])

df.text = df.text.apply(filter_stopwords)

vectorizer = TfidfVectorizer()


word_frequency = vectorizer.fit_transform(df.text)
print('Vektörize edilmiş özelliklerin boyutu:' ,len(vectorizer.get_feature_names()))

sample_index = np.random.random(df.shape[0])

X_train, X_test = word_frequency[sample_index <= TRAIN_PORTION, :], word_frequency[sample_index > TRAIN_PORTION, :]

Y_train, Y_test = df.target[sample_index <= TRAIN_PORTION], df.target[sample_index > TRAIN_PORTION]

df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)
print("TRAIN size - Eğitim Boyutu: ", len(df_train))
print("TEST size - Test Boyutu: ", len(df_test))

clf = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial').fit(X_train, Y_train)
Y_predict = clf.predict(X_test)
Accuracy = sum(Y_predict == Y_test) / len(Y_test)
print("Doğruluk oranı: ",Accuracy)

