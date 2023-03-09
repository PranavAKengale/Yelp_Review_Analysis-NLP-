import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline


import nltk

from nltk.corpus import stopwords

df=pd.read_csv(r'C:\Users\prana\Downloads\Python data science\Refactored_Py_DS_ML_Bootcamp-master\20-Natural-Language-Processing\yelp.csv')

df.head(5)

df.describe()

df.info()

df['text length']=df['text'].apply(len)

df

df['text length'].hist(figsize=(16,8),bins=30)

g=sns.FacetGrid(df,col='stars')
g.map(plt.hist,'text length',bins=50)

plt.figure(figsize=(10,5))
sns.boxplot(x='stars',y='text length',data=df,palette='rainbow')

sns.countplot(x='stars',data=df,palette='rainbow')

stars=df.groupby('stars').mean()

stars

stars.corr()

plt.figure(figsize=(10,5))
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)

df_class=df[(df['stars']==1) | (df['stars']==5)]

df_class.info()

X = df_class['text']
y = df_class['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(X_train,y_train)

predictions = nb.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

# Using  CountVectorizer, TfidfTransformer

import string

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer

#df2=df_class['text'].apply(text_process)

df_class['text']



pipe=Pipeline([('bow',CountVectorizer(analyzer=text_process)),
               ('tfidf',TfidfTransformer()),
               ('model',MultinomialNB())])

from sklearn.model_selection import train_test_split

X=df_class['text']
y=df_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipe.fit(X_train,y_train)

predictions = pipe.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
