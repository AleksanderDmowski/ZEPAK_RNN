# %%
"""
# Recurrent Neural Network
"""

# %%
"""
## Part 1 - Data Preprocessing
"""

# %%
"""
### Importing the libraries
"""

# %%
%reset -f

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %%
from datetime import datetime, timedelta
import tensorflow as tf
from keras.backend import clear_session
tf.compat.v1.reset_default_graph()
clear_session() 

# %%
"""
### Importing the training set
"""

# %%
stock='ZEPAK'
dataset= pd.read_csv('GPW_{}.csv'.format(stock))

iloc1=4
iloc2=5


ostatnia_wartosc = dataset['Dzień'].iloc[-1]
ostatnia_wartosc = pd.to_datetime(ostatnia_wartosc).strftime('%Y-%m-%d')
ostatnia_wartosc= str(ostatnia_wartosc)

delta=10

data= datetime.strptime(ostatnia_wartosc, '%Y-%m-%d')
data= data-timedelta(days = delta)
data= pd.to_datetime(data).strftime('%Y-%m-%d')
print(data)

dataset_train1= dataset[dataset['Dzień'] < data]
training_set1 = dataset_train1.iloc[:,iloc1:iloc2].values
arr = training_set1

# Zmiana typu z string na int
for i in range(len(arr)):
    arr[i] = round(float(arr[i][0].replace(',', '.')), 3)
training_set = arr

dataset_test1 = dataset[dataset['Dzień'] >= data]


print(len(dataset),len(training_set),len(dataset_test1))



# %%
dataset_test1

# %%
data

# %%
ostatnia_wartosc


# %%
"""
### Feature Scaling
"""

# %%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# %%
"""
### Creating a data structure with 60 timesteps and 1 output
"""

# %%
X_train = []
y_train = []
X_train2 = []
timestep=120
for i in range(timestep,  len(training_set)):
    X_train.append(training_set_scaled[i-timestep:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# %%
"""
### Reshaping
"""

# %%
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# %%
"""
## Part 2 - Building and Training the RNN
"""

# %%
"""
### Importing the Keras libraries and packages
"""

# %%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout




# %%
"""
### Initialising the RNN
"""

# %%
regressor = Sequential()

# %%
"""
### Adding the first LSTM layer and some Dropout regularisation
"""

# %%
regressor.add(LSTM(units = 60, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

# %%
"""
### Adding a second LSTM layer and some Dropout regularisation
"""

# %%
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2))

# %%
"""
### Adding a third LSTM layer and some Dropout regularisation
"""

# %%
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2))

# %%
"""
### Adding a fourth LSTM layer and some Dropout regularisation
"""

# %%
regressor.add(LSTM(units = 60))
regressor.add(Dropout(0.2))

# %%
"""
### Adding the output layer
"""

# %%
regressor.add(Dense(units = 1))

# %%
"""
### Compiling the RNN
"""

# %%
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# %%
"""
### Fitting the RNN to the Training set
"""

# %%
regressor.fit(X_train, y_train, epochs =100, batch_size = 32)

# %%
"""
## Part 3 - Making the predictions and visualising the results
"""

# %%
"""
### Getting the real stock price of 2017
"""

# %%
# dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test1.iloc[:, iloc1:iloc2].values
arr = real_stock_price
# Zmiana typu z string na int
for i in range(len(arr)):
    arr[i] = round(float(arr[i][0].replace(',', '.')), 3)
real_stock_price = arr


len(real_stock_price)

# %%
"""
### Getting the predicted stock price of 2017
"""

# %%
dataset_total = pd.concat((dataset_train1['Kurs zamknięcia'], dataset_test1['Kurs zamknięcia']), axis = 0)
arr = dataset_total
# Zmiana typu z string na int
for i in range(len(arr)):
    arr[i] = (float(arr[i].replace(',', '.')))
dataset_total = arr

inputs = dataset_total[len(dataset_total) - len(dataset_test1) - timestep:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(timestep, timestep + len(dataset_test1)):
    X_test.append(inputs[i - timestep:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# X_test_combined = X_test

# # Dopasowanie kształtu danych testowych
# X_test_combined = np.reshape(X_test_combined, (X_test_combined.shape[0], X_test_combined.shape[1], 1))

# Predykcja na danych testowych
predicted_test_prices = regressor.predict(X_test)
predicted_test_prices = sc.inverse_transform(predicted_test_prices)

# print(predicted_stock_price)
# # Predykcja na danych testowych
# # predicted_stock_price = regressor.predict(X_test)
# # predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# %%
# Przewidywanie 10 dni do przodu
X_future = X_test.copy()
predicted_future_prices = []
for i in range(10):
    predicted_price = regressor.predict(X_future)
    predicted_future_prices.append(predicted_price[0, 0])
    X_future = np.roll(X_future, -1)


# Skalowanie odwrotne
predicted_future_prices = sc.inverse_transform(np.array(predicted_future_prices).reshape(-1, 1))

predicted_future_prices=predicted_future_prices.tolist()
# Wyświetlanie przewidywanych wartości
# pfp_zepak_20230711=[[20.34819],
#  [21.669386],
#  [22.290026],
#  [22.327923],
#  [22.439547],
#  [22.188908],
#  [22.116632],
#  [22.198847],
#  [22.482422],
#  [22.736067]]

# pfp_zepak_20230712=[[20.176626],
#  [22.483492],
#  [23.121317],
#  [22.927519],
#  [22.81228 ],
#  [23.133865],
#  [23.50026 ],
#  [23.68236 ],
#  [23.411707],
#  [23.229382]]



# %%
"""
### Visualising the results
"""

# %%
import itertools

ppp=[predicted_test_prices, predicted_future_prices]
# ppp2=[predicted_test_prices[:-1], pfp_zepak_20230711]
# ppp3=[predicted_test_prices[:-1], pfp_zepak_20230712]
ppp=list(itertools.chain.from_iterable(ppp))
# ppp2=list(itertools.chain.from_iterable(ppp2))
# ppp3=list(itertools.chain.from_iterable(ppp3))
# print(type(ppp3))
# print(type(ppp2))
print(type(ppp))


# %%
ostatnia_wartosc

# %%

r=delta+9

lable=[]
ostatnia_wartosc=str(ostatnia_wartosc)
for i in range(r):
    lable.append(i)
    
lables=[]
for i in range(r):
    lables.append('')

lables[0]=str(data)
lables[len(real_stock_price)-1]=str(ostatnia_wartosc)
lables[len(real_stock_price)-2]='D −1'
lables[len(real_stock_price)]='D +1'

ostatnia_wartosc = datetime.strptime(ostatnia_wartosc, '%Y-%m-%d').date()
# lables[len(ppp3)]=(ostatnia_wartosc+timedelta(days=10)).strftime('%Y-%m-%d')

plt.plot(real_stock_price, color = 'red', label = 'True Price')
# plt.plot(ppp3, color = 'grey', linestyle='dashed', alpha=1, label = 'Old D-2 Predicted Stock Price')
#plt.plot(ppp2, color = 'black', linestyle='dashed', alpha=0.2,  label = 'Old D-1 Predicted Stock Price')
plt.plot(ppp, color = 'green', linestyle='dashed', label = 'Future Predicted Stock Price')
plt.plot(predicted_test_prices, color = 'blue', label = 'Test Predicted Stock Price')
# plt.gca().invert_xaxis()
plt.xticks(ticks=lable, labels=lables, rotation=85)
plt.title('{} Stock Price Prediction'.format(stock))
plt.xlabel('Time')
plt.ylabel('{} Stock Price'.format(stock))
plt.legend()

regressor.save(r'C:\Users\Aleksander\Desktop\ZEPAK_RNN\{}_RNN_MODEL.h5'.format(stock))
plt.savefig(r'C:\Users\Aleksander\Desktop\ZEPAK_RNN\{}_prediction.png'.format(stock), bbox_inches="tight")
# plt.show()


# %%
%reset -f