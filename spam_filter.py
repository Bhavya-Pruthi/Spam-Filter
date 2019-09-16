 # -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:01:57 2019

@author: bhavy
"""
import numpy as np
import pandas as pd
data=pd.read_csv("spam.csv")
data['target'] = np.where(data['target']=='spam',1,0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['text'], 
                                                    data['target'], 
                                                    random_state=0)


import operator
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(X_train)
#highest count word
sorted([(token, len(token)) for token in vectorizer.vocabulary_.keys()], key=operator.itemgetter(1), reverse=True)[0]


# multinomial Naive Bayes classifier model  using Count Vector
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)
clf = MultinomialNB(alpha=0.1)
clf.fit(X_train_transformed, y_train)
y_predicted = clf.predict(X_test_transformed)
roc_auc_score(y_test, y_predicted)

#logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_transformed, y_train)
predictions = model.predict(X_test_transformed)
print('AUC: ', roc_auc_score(y_test, predictions))

# multinomial Naive Bayes classifier model  using TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=3)
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)
clf = MultinomialNB(alpha=0.1)
clf.fit(X_train_transformed, y_train)
# y_predicted_prob = clf.predict_proba(X_test_transformed)[:, 1]
y_predicted = clf.predict(X_test_transformed)
roc_auc_score(y_test, y_predicted)


#Returns sparse feature matrix with added feature.feature_to_add can also be a list of features.
def add_feature(X, feature_to_add):
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


#SVC using min dg=5 and added features
from sklearn.svm import SVC
vectorizer = TfidfVectorizer(min_df=5)
X_train_transformed = vectorizer.fit_transform(X_train)    #size=4179x146
X_train_transformed_with_length = add_feature(X_train_transformed, X_train.str.len())  #siz=4179x1469
X_test_transformed = vectorizer.transform(X_test)
X_test_transformed_with_length = add_feature(X_test_transformed, X_test.str.len())
clf = SVC(C=10000)
clf.fit(X_train_transformed_with_length, y_train)
y_predicted = clf.predict(X_test_transformed_with_length)
roc_auc_score(y_test, y_predicted)

#logistic regression with tfidf ,ngrams,min df=5 and added features
vectorizer = TfidfVectorizer(min_df=5, ngram_range=[1,3])
X_train_transformed = vectorizer.fit_transform(X_train)  #size =4179x3383
X_train_transformed_with_length = add_feature(X_train_transformed, [X_train.str.len(),X_train.apply(lambda x: len(''.join([a for a in x if a.isdigit()])))])    #add length and number of digts
X_test_transformed = vectorizer.transform(X_test)
X_test_transformed_with_length = add_feature(X_test_transformed, [X_test.str.len(),X_test.apply(lambda x: len(''.join([a for a in x if a.isdigit()])))])   #size=4179x3385
clf = LogisticRegression(C=100)
clf.fit(X_train_transformed_with_length, y_train)
y_predicted = clf.predict(X_test_transformed_with_length)
roc_auc_score(y_test, y_predicted)

#try with SVC
clf = SVC(C=10000)
clf.fit(X_train_transformed_with_length, y_train)
y_predicted = clf.predict(X_test_transformed_with_length)
roc_auc_score(y_test, y_predicted)

#try with  Naive Bayes 
clf = MultinomialNB(alpha=0.1)
clf.fit(X_train_transformed_with_length, y_train)
y_predicted = clf.predict(X_test_transformed_with_length)
roc_auc_score(y_test, y_predicted)
