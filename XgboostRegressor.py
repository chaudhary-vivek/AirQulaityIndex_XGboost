# In this program, we predict the air quality index uding XGBoost

# Part 1: Preprocessing

# Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the data
df=pd.read_csv('Real_Combine.csv')
# Checking for null values
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Since there are very few null values, we can drop them
df = df.dropna()
# Defining dependent and independent features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Seeing the feature importance using ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)
# Plotting the importance using ExtraTreesRegressor
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

# Splitting the data using train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Part 2: Making XGboost regressor

# Importing Xgboost library
import xgboost as xgb

# Initiating the Xgboost object model
regressor = xgb.XGBRegressor()
#Fitting the data in the model
regressor.fit(X_train, y_train)

# Seeing the coefficient of determination for the training set
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))
# Seeing the coefficient of determination for the testing set
print("Coefficient of determination R^2 <-- on test set: {}".format(regressor.score(X_test, y_test)))
# The coefficient of determination is 0.86 for training and 0.72 for testing. 

# Seeing the crossvalidation score for the model
from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)

# Part 3: Making predictions and evaluating initial model

# Making predictions using the model
prediction=regressor.predict(X_test)
# Plotting the difference of predicted value and the test va;ue
sns.distplot(y_test-prediction)
#Looks normally distributed
# Plotting the predicted values versus the test vvalues
plt.scatter(y_test,prediction)
# Looks somewhat linear

# Part 4: Hyperparameter tuning

# Importing randomized searchcv
from sklearn.model_selection import RandomizedSearchCV

# Defining parameters for hyper parameter tuning
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Various learning rate parameters
learning_rate = ['0.05','0.1', '0.2','0.3','0.5','0.6']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
#Subssample parameter values
subsample=[0.7,0.6,0.8]
# Minimum child weight parameters
min_child_weight=[3,4,5,6,7]

# Creating dictionary to store all the above parameters for hyper parameter tuning
random_grid = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'max_depth': max_depth,
               'subsample': subsample,
               'min_child_weight': min_child_weight}

# Creating the base model to tune
regressor=xgb.XGBRegressor()

# Making random grid search on the base model using the random_grid dictionary with 3 fold cross validation and 100 iterations
xg_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)

# Fitting the data into the grid search
xg_random.fit(X_train,y_train)

# Seeing the best parameters for the model
xg_random.best_params_
# Seeing the best score for the model
xg_random.best_score_
# The score is better then a random forest forest regressor, -1380, compared to -1550

# Part 5: Prediction and evaluation using the tuned model

# Making the prediction for the test set
predictions=xg_random.predict(X_test)

# Plotting the diffrencec of the predicted value and the test value
sns.distplot(y_test-predictions)
# Looks normal with very high kurtosis
# Plotting a scatter of predicted value and the test value
plt.scatter(y_test,predictions)
# Looks linear

# Calculating the MAE, MSE and RMSE

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
# Again, better then random forest.

# Part 6: Dumping the data using pickle for future deployment
 
# Importing pickkle
import pickle

# opening the file to dump the model
file = open('xgboost_regression_model.pkl', 'wb')
# dumping the tuned model into the file
pickle.dump(xg_random, file)





