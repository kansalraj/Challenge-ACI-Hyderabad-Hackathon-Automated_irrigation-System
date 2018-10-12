import pandas as pd
import numpy as np

#importing the dataset
dataset = pd.read_csv('data/data.csv')

dataset=dataset.fillna(0)
dataset=dataset.drop(['Unnamed: 21', 'Unnamed: 22'], axis=1)


#result of independentvariable
X= dataset.iloc[:,[0,1,2,4,9,10,13]].values


#result of dependent variable
y = dataset.iloc[:,-1].values




#y=y.astype(int)

X = np.array(X)
y = np.array(y)





from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor

r_square_list = []

for i in range(1,100):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.10,random_state = i)

    neigh = KNeighborsRegressor(n_neighbors=8)
    neigh.fit(X_train,y_train)

    predicted = neigh.predict(X_test)
    #print("Prediction Result: ",predicted)
    #print("i = ",i)
    r_square = neigh.score(X_test,y_test)
    r_square_list.append(r_square)
    #print('R-squared test score: {:.3f}'.format(neigh.score(X_test_n,y_test_n))) # R-Squared test score

max_r_square = max(r_square_list)
print('Maximum R-squared test score: {:.3f}'.format(max_r_square)) # R-Squared test score







# Feature Scaling KNN

import warnings; warnings.simplefilter('ignore') # Jupyter notebook warning message remove
# Feature Scaling- MinMaxScaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()

r_square_list = []
for i in range(1,100):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.10,random_state = i)
    y_train =y_train.reshape(-1, 1)
    y_test=y_test.reshape(-1, 1)

    # Feature Scaling- MinMaxScaler

    X_train_n = scaler.fit_transform(X_train) # X_train_n = normalized
    X_test_n = scaler.fit_transform(X_test)
    y_train_n = scaler.fit_transform(y_train)
    y_test_n = scaler.fit_transform(y_test)

    neigh = KNeighborsRegressor(n_neighbors=8)
    neigh.fit(X_train_n,y_train_n)

    predicted = neigh.predict(X_test_n)
    #print("Prediction Result: ",predicted)
    #print("i = ",i)
    r_square = neigh.score(X_test_n,y_test_n)
    r_square_list.append(r_square)
    #print('R-squared test score: {:.3f}'.format(neigh.score(X_test_n,y_test_n))) # R-Squared test score

max_r_square = max(r_square_list)
print('Maximum R-squared test score: {:.3f}'.format(max_r_square)) # R-Squared test score

#Undo the scaling of X according to feature_range.    
X_train_d = scaler.inverse_transform(X_train_n) # X_train_d = dnormalized
X_test_d = scaler.inverse_transform(X_test_n)
y_train_d = scaler.inverse_transform(y_train_n)
y_test_d = scaler.inverse_transform(y_test_n)
predicted_d = scaler.inverse_transform(predicted)  


























# Random Forest- Cross Validation
#https://www.kaggle.com/dansbecker/cross-validation
    
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

from sklearn.model_selection import cross_val_score
scores1 = cross_val_score(my_pipeline, X, y, scoring='r2', cv=5)
print(scores1)
print('R^2 score: %2f' %(-1 * scores1.mean()))

scores2 = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores2)
print('Mean Absolute Error %2f' %(-1 * scores2.mean()))

import matplotlib.pyplot as plt
plt.plot(y_test,label='Actual')
plt.plot(predicted1,label='Predicted')
plt.legend()
plt.show()




























# Feature Scaling Random Forest

import warnings; warnings.simplefilter('ignore') # Jupyter notebook warning message remove

# Feature Scaling- MinMaxScaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()


from sklearn.ensemble import RandomForestRegressor

#regressor = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=None, max_features=1, min_samples_leaf=1, min_samples_split=2, bootstrap=False)
r_square_list = []
for i in range(1,100):
    #regressor = RandomForestRegressor(max_depth=None, random_state=i)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.10,random_state = i)
    y_train =y_train.reshape(-1, 1)
    y_test=y_test.reshape(-1, 1)
    # Feature Scaling- MinMaxScaler

    X_train_n = scaler.fit_transform(X_train)
    X_test_n = scaler.fit_transform(X_test)
    y_train_n = scaler.fit_transform(y_train)
    y_test_n = scaler.fit_transform(y_test)

    regressor = RandomForestRegressor(n_estimators=200, max_depth=None, max_features=1, min_samples_leaf=1, min_samples_split=2, bootstrap=False)
    regressor.fit(X_train_n, y_train_n)

    predicted = regressor.predict(X_test_n)
    #print("Prediction Result: ",predicted)
    #print("i = ",i)

    #print('R-squared test score: {:.3f}'.format(regressor.score(X_test_n,y_test_n))) # R-Squared test score
    r_square = regressor.score(X_test_n,y_test_n)
    r_square_list.append(r_square)

max_r_square = max(r_square_list)
print('Maximum R-squared test score: {:.3f}'.format(max_r_square)) # R-Squared test score

#Undo the scaling of X according to feature_range.    
X_train_d = scaler.inverse_transform(X_train_n)
X_test_d = scaler.inverse_transform(X_test_n)
y_train_d = scaler.inverse_transform(y_train_n)
y_test_d = scaler.inverse_transform(y_test_n)
predicted_d = scaler.inverse_transform(predicted)















































from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.10,random_state = 100)

# nn = MLPRegressor(
#     hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
#     learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
#     random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn = MLPRegressor(hidden_layer_sizes=(10, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', 
                  learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, 
                  random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                  early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn.fit(X_train, y_train)
pred=nn.predict(X_test)

r_square = nn.score(X_test,y_test)
print('R-squared test score:',r_square)

# calculate MAE
mae = mean_absolute_error(y_test, pred)
print('Test MAE: %.3f' % mae)

# calculate MSE
mse = mean_squared_error(y_test, pred)
print('Test MSE: %.3f' % mse)

# calculate RMSE
rmse = sqrt(mean_squared_error(y_test, pred))
print('Test RMSE: %.3f' % rmse)







































# NN- Cross Validation
from sklearn.neural_network import MLPRegressor

# nn = MLPRegressor(
#     hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
#     learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
#     random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn = MLPRegressor(hidden_layer_sizes=(10, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', 
                  learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, 
                  random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                  early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn.fit(X_train, y_train)
pred=nn.predict(X_test)

r_square = nn.score(X_test,y_test)
print(r_square)





























# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.metrics import mean_absolute_error

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.10,random_state = 40)

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))

# calculate MAE
mae = mean_absolute_error(y_test, yhat)
print('Test MAE: %.3f' % mae)

# calculate MSE
mse = mean_squared_error(y_test, yhat)
print('Test MSE: %.3f' % mse)

# calculate RMSE
rmse = sqrt(mean_squared_error(y_test, yhat))
print('Test RMSE: %.3f' % rmse)