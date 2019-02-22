
# coding: utf-8

# In[2]:

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns


# Import the dataset
dataset = pd.read_csv('train.csv')
full_dataset = dataset.iloc[:, :].values
X = full_dataset[:, :-1]
y = full_dataset[:, 12]
y = y > 0
# https://www.python-course.eu/numpy_masking.php
# convert the array to a normal int instead of boolean
y = y.astype(np.int)

dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, :-1].values
y_test = dataset_test.iloc[:, 12].values
y_test = y_test > 0
y_test = y_test.astype(np.int)

# plot the heatmap showing correlation among features
corr = dataset.corr()
fig = plt.subplots(figsize = (10,10))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.show()


# In[3]:

X


# In[4]:

X.shape


# In[5]:

full_dataset.shape



# In[6]:

y.shape


# In[7]:

col_list = list(dataset)
for feature in range(0,12,1):
    plt.title('Histogram of input feature')
    plt.hist(X[:,feature])
    plt.xlabel(col_list[feature])
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()


# In[8]:

# Feature Standardization
import sklearn 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)
# This will use the training set values for the Standardization
X_test = sc_x.transform(X_test)


# In[9]:

X[0] 


# In[10]:

X_test[0]


# In[11]:

y_test


# In[12]:

"""
col_list = list(dataset)
for feature in range(0,12,1):
    plt.title('Histogram of input feature')
    plt.hist(X[:,feature])
    plt.xlabel(col_list[feature])
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
"""    


# In[13]:

def KNNClassifier(K, X, y, test_samples):
    y_copy = np.copy(y)
    y_copy = np.reshape(y_copy,(X.shape[0], 1))
    TrainingData = np.append(X, y_copy, axis=1)
    prediction=[]
    for test_sample in test_samples:
        sorted_neighbours =        sorted(TrainingData,key=               lambda               Trainingsample:               np.linalg.norm(Trainingsample[0:12]-test_sample))
        
        sorted_neighbours = np.asarray(sorted_neighbours)
        k_neighbors = sorted_neighbours[0:K, 12]
        unique, counts = np.unique(k_neighbors, return_counts=True)
        pred_map = dict(zip(unique, counts))
        
        if 0.0 not in pred_map:
            pred_map[0.0] = 0
        
        if 1.0 not in pred_map:
            pred_map[1.0] = 0
            
        if (pred_map[0.0]>pred_map[1.0]):
            prediction.append(0)
        else:
            prediction.append(1)
    return np.array(prediction)
    


# In[14]:

# K-fold validation
# code referenced from this book - https://www.manning.com/
#                           books/deep-learning-with-python
k = 9
train_data = X
train_targets = y
num_val_samples = train_data.shape[0] // k
accuracy = []
for knn in range(1,52,2):
    all_scores = []
    for i in range(k):
        val_data =        train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets =        train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)

        val_pred =        KNNClassifier(knn, partial_train_data, partial_train_targets, val_data)
        acc = accuracy_score(val_targets, val_pred)
        all_scores.append(acc)
    accuracy.append(np.mean(all_scores))    



# In[15]:

print (accuracy)

epochs = range(1, 52, 2)    
plt.plot(epochs, accuracy, 'b', label='Accuracy cross validation')
plt.title('Hyper Parameter tuning')
plt.xlabel('value of knn - K')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

A = epochs
B = accuracy

plt.plot(A,B,'b', label='Accuracy cross validation')
for xy in zip(A, B):                                       # <--
    ax.annotate('(%s, %.2f)' % xy, xy=xy, textcoords='data') # <--

plt.title('Hyper Parameter tuning')
plt.xlabel('value of knn - K')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

print('Highest Accuracy is {}'.format(accuracy[-5]))


# In[16]:

# Making Predictions
y_pred = KNNClassifier(43, X, y, X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)
# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[17]:

# Trying other metrics
# https://scikit-learn.org/stable/modules/
# generated/sklearn.neighbors.DistanceMetric.html

# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=43, metric='manhattan')
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

accuracy_score(y_test, y_pred)


# In[18]:

classifier = KNeighborsClassifier(n_neighbors=43, metric='chebyshev')
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

accuracy_score(y_test, y_pred)


# In[19]:

classifier = KNeighborsClassifier(n_neighbors=43, metric='hamming')
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

accuracy_score(y_test, y_pred)


# In[20]:

classifier = KNeighborsClassifier(n_neighbors=43, metric='braycurtis')
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

accuracy_score(y_test, y_pred)


# In[21]:

classifier = KNeighborsClassifier(n_neighbors=43, metric='jaccard')
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

accuracy_score(y_test, y_pred)


# In[22]:

# Custome metric = hamming distance for categorical data 
# and euclidean for real valued
from sklearn.neighbors import DistanceMetric
def mydist(x, y):
    euclidean = np.sum((x[4:]-y[4:])**2)
    hamming = sum(ch1 != ch2 for ch1, ch2 in zip(x[0:4], y[0:4]))
    return euclidean + hamming

classifier = KNeighborsClassifier(n_neighbors=43, metric=mydist)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

accuracy_score(y_test, y_pred)


# In[23]:

# Custome metric = similarity(opposite of hamming) 
# for categorical data and euclidean for real valued
from sklearn.neighbors import DistanceMetric
def mydist(x, y):
    euclidean = np.sum((x[4:]-y[4:])**2)
    similarity = sum(ch1 == ch2 for ch1, ch2 in zip(x[0:4], y[0:4]))
    return euclidean + similarity

classifier = KNeighborsClassifier(n_neighbors=43, metric=mydist)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

accuracy_score(y_test, y_pred)


# In[ ]:



