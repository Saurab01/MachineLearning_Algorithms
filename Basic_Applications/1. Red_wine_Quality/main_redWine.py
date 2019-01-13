import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split #to split our dataset into training
# and testing data,

from sklearn import preprocessing   #to preprocess the data before fitting into predictor,
# or converting it to a range of -1,1, which is easy to understand for the machine learning algorithms.

from sklearn import tree  #to import our decision tree classifier,
# which we will be using for prediction.

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

#print (data.head())   #get first 5 values
print ("main data shape::",data.shape)

'''Now, in every machine learning program, there are two things, features and labels. 
Features are the part of a dataset which are used to predict the label. 
And labels on the other hand are mapped to features. 

After the model has been trained, we give features to it, so that it can predict the labels.
So, if we analyse this dataset, since we have to predict the wine quality, 
the attribute quality will become our label and the rest of the attributes will become the 
features.
'''

#separate the features and labels into two different dataframes.
y = data.quality
X = data.drop('quality', axis=1)  #quality dropped

# slpitting into train and test
#test_size=0.2 to make the test data 20% of the original data. # The rest 80% is used for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print ("\nX_train:\n",X_train.head())
print ("X_train shape: ",X_train.shape)
print ("X_test shape: ",X_test.shape)

#After we obtained the data we will be using, the next step is data normalization.
#  It is part of pre-processing in which data(X) is converted to fit in a range of -1 and 1
X_train_scaled = preprocessing.scale(X_train)
print("\nAfter preprocessing: \n", X_train_scaled)
print ("X_train_scaled.shape:: ",X_train_scaled.shape)

# Using Decision Tree Classifier
clf = tree.DecisionTreeClassifier()

# Fitting: Training the ML Algo
clf.fit(X_train, y_train)

# Obtaining the confidence score for SVR
confidence = clf.score(X_test, y_test)
print("\nThe confidence score::",confidence)
#this result can change ,always expect a range of ±5 around your first result.

# predicting the forcasts
y_pred = clf.predict(X_test)

print ("Predicted y_pred shape==",y_pred.shape)
# printing fthe prediction
print("\nThe prediction:: ",y_pred)

# printing the labeled result expectation
print("\nThe expectation:",y_test)

# converting the numpy array to list
x = np.array(y_pred).tolist()

# printing first 5 predictions
print("\nThe prediction:\n")
for i in range(0, 5):
    print (x[i])

# printing first five expectations
print("\nThe expectation:\n")
print (y_test.head())
