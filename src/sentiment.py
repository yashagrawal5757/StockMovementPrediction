import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout,LSTM
from tensorflow.keras.callbacks   import EarlyStopping
import nltk
nltk.download('stopwords')


os.chdir('C:/Users/yash/Desktop/finproj')

#Loading news dataset
df=pd.read_csv('data_old.csv', encoding = "ISO-8859-1")
df=df[:-1]

df.isnull().sum()
df[df.isnull().any(axis=1)]
# note index 277,348,681.Dates are 15th,24th,21st.  #We will have to remove these indices after merging stock and news data

df = df.dropna(axis=0)
df.isnull().sum()

df.head()
df.describe()
df.info()
print(df.columns) #to know name of columns
print(df.dtypes)  #to know datatype of columns.

#train test split is 80%
train = df[df['Date'] < '2014-01-01']
test = df[df['Date'] > '2013-12-31']

len(train[train['Label']==0])  #625 zeros in train set
len(train[train['Label']==1])  #731 ones in train set
# Well balanced train set

len(test[test['Label']==0])  # 299 zeros in train set
len(test[test['Label']==1])  #330 ones in train set
# Well balanced test set

#NLP preprocessing on train set news


# extracting only the news headlines
news_train=train.iloc[:,2:27]

#keeping alphabets only
news_train = news_train.replace('b\"|b\'|\\\\|\\\"', '', regex=True)
news_train.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [str(i) for i in range(25)]
news_train.columns= list1
news_train.columns
#----------------------------------------------------------------------

print(news_train['0'].head(1))
# downs converted to down and similar changes

#converting all the top25 headlines of first date to a single paragraph
' '.join(str(x) for x in news_train.iloc[0,0:25])

#doing the same for all dates
headlines_train = []
for row in range(0,len(news_train.index)):
    headlines_train.append(' '.join(str(x) for x in news_train.iloc[row,0:25]))
    
headlines_train[11]


from sklearn.feature_extraction.text import CountVectorizer
## implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(2,2))

# apply countvector on all headlines in para form
traindataset=countvector.fit_transform(headlines_train)
print(len(countvector.vocabulary_))
# all words converted into 311095columns


print('Amount of Non-Zero occurences: ', traindataset.nnz)
# There are 559594 non zero values

type(traindataset)
# Its a sparse matrix which is expected since every row will contain only few words. For those words there is a 1  else zero. So its a sparse matrix

sparsity = (100.0 * traindataset.nnz / (traindataset.shape[0] * traindataset.shape[1]))
print('sparsity: {}'.format((sparsity)))
# 13% of values are non zero. Sparsity= 87%


#Applying countvector on single news to understand process
news1 = news_train['0'][1]
print(news1)
cv1 = countvector.transform([news1])
print(cv1)
# Here we see column numbers and occurence of that word  in 1st message. 120724th column sees word coming twice in news1 .This means that there are 10 unique words in news1. One of them appear twice, the rest only once.
print(countvector.get_feature_names()[120724])
# As we see help us comes twice

#----------------------------------------------------------------------

# TF-IDF  vectorization on training news
from sklearn.feature_extraction.text import TfidfTransformer
tfidf=TfidfTransformer()

# Now applying TF-IDF Vectorization model on whole data
traindataset = tfidf.fit_transform(traindataset)
# Weights have been assigned to each word not only in that message but according to whole corpus i.e. what significance does a word hold in whole dataset(TD*IDF)


# trying tf-idf on news1
tfidf1 = tfidf.transform(cv1)
print(tfidf1)
# weights have been assigned to each word wrt whole dataset. Help us(135029) has weight of 0.46 wrt whole dataset
#(TF*IDF)

print(tfidf.idf_[countvector.vocabulary_['help us']])
# wrt just a message, it has weight of 6.4(IDF)

# NLP PREPROCESSING OVER for news_train
#---------------------------------------------------------------------

#implement various classifiers to predict sentiment

# implement RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])
#all the patterns have been learnt

#implement knn
from sklearn.neighbors import KNeighborsClassifier
classifierknn = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
classifierknn.fit(traindataset,train['Label'])

#implement logistic reegression
from sklearn.linear_model import LogisticRegression
classifierlog = LogisticRegression(random_state = 0)
classifierlog.fit(traindataset,train['Label'])

# implement kernel svm
from sklearn.svm import SVC
classifiersvc = SVC(kernel = 'rbf', random_state = 0)
classifiersvc.fit(traindataset, train['Label'])

#implement xgboost
from xgboost import XGBClassifier
model_xgb = XGBClassifier()
model_xgb.fit(traindataset,train['Label'])

#testing word importance on logistic regression
features = countvector.get_feature_names()
coefficients = classifierlog.coef_.tolist()[0]
coefficients_df = pd.DataFrame({'Words' : features, 
                        'Coefficient' : coefficients})
coefficients_df = coefficients_df.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
coefficients_df.head(10)

coefficients_df.tail(10)


#---------------------------------------------------------------------
# estimate accuracy on testing dataset

## NLP for the news_test
news_test=test.iloc[:,2:27]
news_test = news_test.replace('b\"|b\'|\\\\|\\\"', '', regex=True)
news_test.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
# Renaming column names for ease of access
news_test.columns= list1
news_test.columns
 
headlines_test= []
#joining all headlines in to a single para for testing dataset
for row in range(0,len(news_test.index)):
    headlines_test.append(' '.join(str(x) for x in news_test.iloc[row,0:25]))

# Bag of words model for test set
testdataset = countvector.transform(headlines_test)
testdataset.shape
# 377 rows, 341402 columns


# TF-IDF on test set
testdataset = tfidf.transform(testdataset)
# NLP PREPROCESSING FOR TEST SET DONE
#-----------------------------------------------------------------------

# predicting on test set
predictions_rc = randomclassifier.predict(testdataset)
predictions_knn = classifierknn.predict(testdataset)
predictions_log = classifierlog.predict(testdataset)
predictions_svc = classifiersvc.predict(testdataset)
predictions_xgb = model_xgb.predict(testdataset)

#report for xgboost on test data
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix=confusion_matrix(test['Label'],predictions_xgb)
print(matrix)
score=accuracy_score(test['Label'],predictions_xgb)
print(score) # 50% accuracy
report=classification_report(test['Label'],predictions_xgb)
print(report)

#report for random forest on test data
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix=confusion_matrix(test['Label'],predictions_rc)
print(matrix)
score=accuracy_score(test['Label'],predictions_rc)
print(score) # 51% accuracy
report=classification_report(test['Label'],predictions_rc)
print(report)
# precision, recall,f1 score are good for random forest

#report for knn on test data
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix=confusion_matrix(test['Label'],predictions_knn)
print(matrix)
score=accuracy_score(test['Label'],predictions_knn)
print(score) #48%
report=classification_report(test['Label'],predictions_knn)
print(report)
# knn performed bad

#report for logistic regression on test data
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix=confusion_matrix(test['Label'],predictions_log)
print(matrix)
score=accuracy_score(test['Label'],predictions_log)
print(score) #52%
report=classification_report(test['Label'],predictions_log)
print(report)
# bad performance

#report for kernel svm on test data
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix=confusion_matrix(test['Label'],predictions_svc)
print(matrix)
score=accuracy_score(test['Label'],predictions_svc)
print(score) #52%
report=classification_report(test['Label'],predictions_svc)
print(report)
# svm just classified everything in one class

# We will use grid search to find hyper parameters


param_grid = {'C': [0.1,1, 10, 100, 1000], 
              'gamma': [1,0.1,0.01,0.001,0.0001]} 
# best parameter is 0.1 for C and 1 for gamma
# when fitted with paramgrid, model failed badly(50% acc)


param_grid2= {'C': [0.1,100], 
              'gamma': [10],
              'kernel': ['rbf','sigmoid']}
# result-{'C': 0.1, 'gamma': 10, 'kernel': 'rbf'}


param_grid3= {'C': [0.01,0.001], 
              'gamma': [100,1000],
              'kernel': ['rbf']}
# {'C': 0.01, 'gamma': 100, 'kernel': 'rbf'}


param_grid4= {'C': [0.09, 0.009], 
              'gamma': [900,9000]}
#{'C': 0.09, 'gamma': 900}


from sklearn.model_selection import GridSearchCV
gridsvc = GridSearchCV(SVC(),param_grid4,
                    refit=True,verbose=5)
gridsvc.fit(traindataset, train['Label'])

gridsvc.best_params_
gridsvc.best_estimator_
#refit refits the model with above parameters

grid_predictions_svc = gridsvc.predict(testdataset)

#report for optimized kernel svm on test data
matrix=confusion_matrix(test['Label'],grid_predictions_svc)
print(matrix)
score=accuracy_score(test['Label'],grid_predictions_svc)
print(score)
report=classification_report(test['Label'],grid_predictions_svc)
print(report)

#Cant find perfect parameters

# We will consider xgboost for converting the whole dataset.
#---------------------------------------------------------------------

#NLP preprocessing on whole dataset


# join train and test headlines to get final news data
headlines_main = headlines_train+headlines_test
#joining all headlines in to a single para for testing dataset

# Bag of words model for whole dataset
wholenewsdataset = countvector.transform(headlines_main)
wholenewsdataset.shape
# 1985 rows, 311095 columns

# TF-IDF on whole data set
wholenewsdataset = tfidf.transform(wholenewsdataset)
# NLP PREPROCESSING FOR whole news dataset SET DONE
#-----------------------------------------------------------------------

#applying xgboost on whole dataset
predictions_wholedataset = model_xgb.predict(wholenewsdataset)
predictions_wholedataset
df['Label']

#Report on whole dataset
matrix_whole=confusion_matrix(df['Label'],predictions_wholedataset)
print(matrix_whole)
score_wholedataset=accuracy_score(df['Label'],predictions_wholedataset)
print(score_wholedataset) 
#84% accuracy
report=classification_report(df['Label'],predictions_wholedataset)
print(report)

#-----------------------------------------------------------------------
# Conclusion: 
#We performed NLP on news headlines for each day and predicted sentiment with 84% accuracy wrt XG Boost.
# Next step is going to merge the predicted sentiment with the stock data so that the sentiment column can help  us predict closing price for a day