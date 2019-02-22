
# coding: utf-8

# In[2]:

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Import the dataset
dataset = pd.read_csv('train.csv')
full_dataset = dataset.iloc[:, :].values
print('Original Training Sample size : {}'.format(full_dataset.shape))
# get data for which area > 0
new_full_dataset_DF = dataset[dataset.area > 0]
new_full_dataset = new_full_dataset_DF.values
print('Training Sample size (with area > 0) : {}'.format(new_full_dataset.shape))
X = new_full_dataset[:, :-1]
y = new_full_dataset[:, 12]

dataset_test = pd.read_csv('test.csv')
full_test_dataset = dataset_test[dataset_test.area > 0].values
print('Test Sample size (with area > 0) : {}'.format(full_test_dataset.shape))
X_test = full_test_dataset[:, :-1]
y_test = full_test_dataset[:, 12]

# plot the heatmap showing correlation among features
corr = new_full_dataset_DF.corr()
fig = plt.subplots(figsize = (10,10))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.show()

# NOTE : If a warning comes then just run this cell again, 
# it's known error in the library


# In[3]:

# Plotting the graphs
import matplotlib.pyplot as plt

plt.hist(y, bins=10)
plt.title('Histogram of outcome variable')
plt.xlabel('Value of area')
plt.ylabel('Frequency')
plt.grid()
plt.show()


plt.hist(np.log(y))
plt.title('Histogram of outcome variable')
plt.xlabel('Value of log(area)')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[4]:

print('True labels : {}'.format(y))


# In[5]:

print('True labels(log y) : {}'.format(np.log(y)))


# In[6]:

print('Feature Shape : {}'.format(X.shape))


# In[7]:

print('True Label Shape : {}'.format(y.shape))



# In[8]:

X[0]


# In[9]:

# Feature Standardization
import sklearn 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)
X_test = sc_x.transform(X_test)


# In[10]:

X[0] 


# In[11]:

X_test[0]


# In[12]:

y_test


# In[13]:

# Regression
# Fitting Multiple Linear Regression to the Training set
"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


print(regressor.score(X, y))
print(regressor.coef_)
print(np.corrcoef(y_pred, y_test))
print(np.correlate(y_pred, y_test))
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
"""      


# In[14]:

# Regression
# Using OLS to compute the weights
X = np.hstack([np.ones([X.shape[0],1]), X])

a = np.matmul(X.T, X)
a = np.linalg.inv(a)
b = np.matmul(X.T, y)
w = np.matmul(a, b)

print('Shape of weight vector : {}'.format(w.shape))
print('Computed weight vector : {}'.format(w))


# In[15]:

# Prediction
X_test = np.hstack([np.ones([X_test.shape[0],1]), X_test])
y_pred = X_test.dot(w)
print(np.corrcoef(y_pred, y_test))
print(np.cov(y_pred, y_test))
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
RSS = sum((y_pred-y_test)**2)
print("Residual square error(RSS): %.2f"
      % RSS)
print(stats.pearsonr(y_pred, y_test))


# In[16]:

y_pred


# In[17]:

y_test


# In[18]:

# using log scale
# Using OLS to compute the weights with log of area

a = np.matmul(X.T, X)
a = np.linalg.inv(a)
y_log = np.log(y)
b = np.matmul(X.T, y_log)
w = np.matmul(a, b)

print(w.shape)
print(w)

# Prediction
y_pred = X_test.dot(w)
print(np.corrcoef(y_pred, np.log(y_test)))
print(np.cov(y_pred, np.log(y_test)))
print("Mean squared error: %.2f"
      % mean_squared_error(np.log(y_test), y_pred))
RSS = sum((y_pred-np.log(y_test))**2)
print("Residual square error(RSS): %.2f"
      % RSS)
print(stats.pearsonr(y_pred, np.log(y_test)))


# In[19]:

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print(X.shape)
print(X_poly.shape)


# In[20]:

# Predicting the Test set results
X_test_poly = poly_reg.transform(X_test)
y_pred = lin_reg_2.predict(X_test_poly)

print(np.corrcoef(y_pred, y_test))
print(np.cov(y_pred, y_test))
print("Test Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
RSS = sum((y_pred-y_test)**2)
print("Test Residual square error(RSS): %.2f"
      % RSS)

y_train_predict = lin_reg_2.predict(X_poly)
print("Train Mean squared error: %.2f"
      % mean_squared_error(y, y_train_predict))
RSS = sum((y_train_predict-y)**2)
print("Train Residual square error(RSS): %.2f"
      % RSS)
print("Training coreleation coefficient:")
print(stats.pearsonr(y, y_train_predict))


# In[21]:

y_test


# In[22]:

y_pred


# In[23]:

y_pred.shape


# In[24]:

# Fitting Polynomial Regression to the dataset using log of area
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, np.log(y))
# Predicting the Test set results
X_test_poly = poly_reg.transform(X_test)
y_pred = lin_reg_2.predict(X_test_poly)

print(np.corrcoef(y_pred, np.log(y_test)))
print(np.cov(y_pred, np.log(y_test)))
print("Mean squared error: %.2f"
      % mean_squared_error(np.log(y_test), y_pred))
RSS = sum((y_pred-np.log(y_test))**2)
print("Residual square error(RSS): %.2f"
      % RSS)

y_train_predict = lin_reg_2.predict(X_poly)
print("Train Mean squared error: %.2f"
      % mean_squared_error(np.log(y), y_train_predict))
RSS = sum((y_train_predict-np.log(y))**2)
print("Train Residual square error(RSS): %.2f"
      % RSS)
print("Training coreleation coefficient:")
print(stats.pearsonr(np.log(y), y_train_predict))


# In[25]:

cube_reg = PolynomialFeatures(degree=3)
X_cube = cube_reg.fit_transform(X)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_cube, y)

# Predicting the Test set results
X_test_cube = cube_reg.transform(X_test)
y_pred = lin_reg_3.predict(X_test_cube)

print(np.corrcoef(y_pred, y_test))
print(np.cov(y_pred, y_test))
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
RSS = sum((y_pred-y_test)**2)
print("Residual square error(RSS): %.2f"
      % RSS)

y_train_predict = lin_reg_3.predict(X_cube)
print("Train Mean squared error: %.2f"
      % mean_squared_error(y, y_train_predict))
RSS = sum((y_train_predict-y)**2)
print("Train Residual square error(RSS): %.2f"
      % RSS)
print("Training coreleation coefficient:")
print(stats.pearsonr(y, y_train_predict))


# In[26]:

# Fitting Polynomial Regression to the dataset using log of area
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_cube, np.log(y))

# Predicting the Test set results
X_test_cube = cube_reg.transform(X_test)
y_pred = lin_reg_3.predict(X_test_cube)

print(np.corrcoef(y_pred, np.log(y_test)))
print(np.cov(y_pred, np.log(y_test)))
print("Test Mean squared error: %.2f"
      % mean_squared_error(np.log(y_test), y_pred))
RSS = sum((y_pred-np.log(y_test))**2)
print("Test Residual square error(RSS): %.2f"
      % RSS)

y_train_predict = lin_reg_3.predict(X_cube)
print("Train Mean squared error: %.2f"
      % mean_squared_error(np.log(y), y_train_predict))
RSS = sum((y_train_predict-np.log(y))**2)
print("Train Residual square error(RSS): %.2f"
      % RSS)
print("Training coreleation coefficient:")
print(stats.pearsonr(np.log(y), y_train_predict))


# In[ ]:




# In[ ]:



