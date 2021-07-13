import numpy as np # linear algebra
import pandas as pd
import numpy as np
import re
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#veri setindeki verilerin çekilmesi
cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv('dataset/sentiment140.csv', encoding='latin1', names=cols)
df.info()
df['sentiment'].value_counts()
df.head()
pat1 = '@[^ ]+'
pat2 = 'http[^ ]+'
pat3 = 'www.[^ ]+'
pat4 = '#[^ ]+'
pat5 = '[0-9]'

combined_pat = '|'.join((pat1, pat2, pat3, pat4, pat5))
clean_tweet_texts = []
for t in df['text']:
    t = t.lower()
    stripped = re.sub(combined_pat, '', t)
    tokens = word_tokenize(stripped)
    words = [x for x in tokens if len(x) > 1]
    sentences = " ".join(words)
    negations = re.sub("n't", "not", sentences)

    clean_tweet_texts.append(negations)
clean_df = pd.DataFrame(clean_tweet_texts, columns=['text'])
clean_df['sentiment'] = df['sentiment'].replace({4:1})
clean_df.head()
clean_df.info()
neg_tweets = clean_df[clean_df['sentiment']==0]
pos_tweets = clean_df[clean_df['sentiment']==1]
#
# cv = CountVectorizer(stop_words='english', binary=False, ngram_range=(1,1))
#
# neg_cv = cv.fit_transform(neg_tweets['text'].tolist())
# pos_cv = cv.fit_transform(pos_tweets['text'].tolist())
#
# freqs_neg = zip(cv.get_feature_names(), neg_cv.sum(axis=0).tolist()[0])
# freqs_pos = zip(cv.get_feature_names(), pos_cv.sum(axis=0).tolist()[0])
#
# list_freq_neg = list(freqs_neg)
# list_freq_pos = list(freqs_pos)
#
# list_freq_neg.sort(key=lambda tup: tup[1], reverse=True)
# list_freq_pos.sort(key=lambda tup: tup[1], reverse=True)
#
# cv_words_neg = [i[0] for i in list_freq_neg]
# cv_counts_neg = [i[1] for i in list_freq_neg]
# cv_words_pos = [i[0] for i in list_freq_pos]
# cv_counts_pos = [i[1] for i in list_freq_pos]
#
# plt.bar(cv_words_neg[0:10], cv_counts_neg[0:10])
# plt.xticks(rotation='vertical')
# plt.title('Top Negative Words With CountVectorizer')
# plt.show()
# plt.bar(cv_words_pos[0:10], cv_counts_pos[0:10])
# plt.xticks(rotation='vertical')
# plt.title('Top Positive Words With CountVectorizer')
# plt.show()
#  TF-IDF Solution

tv = TfidfVectorizer(stop_words='english', binary=False, ngram_range=(1,3))

neg_tv = tv.fit_transform(neg_tweets['text'].tolist())
pos_tv = tv.fit_transform(pos_tweets['text'].tolist())
freqs_neg_tv = zip(tv.get_feature_names(), neg_tv.sum(axis=0).tolist()[0])
freqs_pos_tv = zip(tv.get_feature_names(), pos_tv.sum(axis=0).tolist()[0])
list_freq_neg_tv = list(freqs_neg_tv)
list_freq_pos_tv = list(freqs_pos_tv)
list_freq_neg_tv.sort(key=lambda tup: tup[1], reverse=True)
list_freq_pos_tv.sort(key=lambda tup: tup[1], reverse=True)

cv_words_neg_tv = [i[0] for i in list_freq_neg_tv]
cv_counts_neg_tv = [i[1] for i in list_freq_neg_tv]

cv_words_pos_tv = [i[0] for i in list_freq_pos_tv]
cv_counts_pos_tv = [i[1] for i in list_freq_pos_tv]
plt.bar(cv_words_neg_tv[0:10], cv_counts_neg_tv[0:10])
plt.xticks(rotation='vertical')
plt.title('Top Negative Words With tf-idf')
plt.show()

plt.bar(cv_words_pos_tv[0:10], cv_counts_pos_tv[0:10])
plt.xticks(rotation='vertical')
plt.title('Top Positive Words with tf-idf')
plt.show()
x = clean_df['text']
y = clean_df['sentiment']
x_tv = tv.fit_transform(x)
x_train_tv, x_test_tv, y_train_tv, y_test_tv = train_test_split(x_tv, y, test_size=0.2, random_state=0)
log_tv = LogisticRegression()
log_tv.fit(x_train_tv,y_train_tv)
y_pred_tv = log_tv.predict(x_test_tv)

print(confusion_matrix(y_test_tv,y_pred_tv))
print(classification_report(y_test_tv,y_pred_tv))
