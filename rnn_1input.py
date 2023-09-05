import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from datetime import datetime, timedelta

import tensorflow as tf

from keras.backend import clear_session

tf.compat.v1.reset_default_graph()

clear_session() 





import os

cwd = os.getcwd()

os.chdir(os.getcwd())
stock='ZEPAK'

dataset= pd.read_csv('{}\GPW_{}.csv'.format(cwd, stock))

iloc1=4

iloc2=5

delta=10



ostatnia_wartosc = dataset['Dzień'].iloc[-1]

dni= pd.DataFrame()

dni["Dzień"] = pd.to_datetime(dataset["Dzień"].iloc[-30:]).dt.strftime('%Y-%m-%d')

dni=dni["Dzień"].values.tolist()



ostatnia_wartosc = pd.to_datetime(ostatnia_wartosc).strftime('%Y-%m-%d')

ostatnia_wartosc= str(ostatnia_wartosc)







data= datetime.strptime(ostatnia_wartosc, '%Y-%m-%d')

data= data-timedelta(days = delta)

data= pd.to_datetime(data).strftime('%Y-%m-%d')



dataset_train1= dataset[dataset['Dzień'] < data]

training_set1 = dataset_train1.iloc[:,iloc1:iloc2].values

arr = training_set1



for i in range(len(arr)):

    arr[i] = round(float(arr[i][0].replace(',', '.')), 3)

training_set = arr



dataset_test1 = dataset[dataset['Dzień'] >= data]







dataset_test1
data
ostatnia_wartosc

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set)

X_train = []

y_train = []

X_train2 = []

timestep=120

for i in range(timestep,  len(training_set)):

    X_train.append(training_set_scaled[i-timestep:i, 0])

    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout





regressor = Sequential()
regressor.add(LSTM(units = 60, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))

regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 60, return_sequences = True))

regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 60, return_sequences = True))

regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 60))

regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs =100, batch_size = 32)
real_stock_price = dataset_test1.iloc[:, iloc1:iloc2].values

arr = real_stock_price

for i in range(len(arr)):

    arr[i] = round(float(arr[i][0].replace(',', '.')), 3)

real_stock_price = arr





len(real_stock_price)
dataset_total = pd.concat((dataset_train1['Kurs zamknięcia'], dataset_test1['Kurs zamknięcia']), axis = 0)

arr = dataset_total

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







predicted_test_prices = regressor.predict(X_test)

predicted_test_prices = sc.inverse_transform(predicted_test_prices)



X_future = X_test.copy()

predicted_future_prices = []

for i in range(10):

    predicted_price = regressor.predict(X_future)

    predicted_future_prices.append(predicted_price[0, 0])

    X_future = np.roll(X_future, -1)





predicted_future_prices = sc.inverse_transform(np.array(predicted_future_prices).reshape(-1, 1))



predicted_future_prices=predicted_future_prices.tolist()





import itertools



ppp=[predicted_test_prices, predicted_future_prices]

ppp=list(itertools.chain.from_iterable(ppp))

print(type(ppp))

ostatnia_wartosc


r=delta+9



lable=[]

ostatnia_wartosc=str(ostatnia_wartosc)

for i in range(r):

    lable.append(i)

    

lables=[]

for i in range(r):

    lables.append('')



lables[0]=str(dni[-delta])

lables[len(real_stock_price)-1]=str(ostatnia_wartosc)

lables[len(real_stock_price)-2]='D −1'

lables[len(real_stock_price)]='D +1'



ostatnia_wartosc = datetime.strptime(ostatnia_wartosc, '%Y-%m-%d').date()



plt.plot(real_stock_price, color = 'red', label = 'True Price')

plt.plot(ppp, color = 'green', linestyle='dashed', label = 'Future Predicted Stock Price')

plt.plot(predicted_test_prices, color = 'blue', label = 'Test Predicted Stock Price')

plt.xticks(ticks=lable, labels=lables, rotation=85)

plt.title('{} Stock Price Prediction'.format(stock))

plt.xlabel('Time')

plt.ylabel('{} Stock Price'.format(stock))

plt.legend()



regressor.save(r'C:\Users\Aleksander\Desktop\ZEPAK_RNN\{}_RNN_MODEL.h5'.format(stock))

plt.savefig(r'C:\Users\Aleksander\Desktop\ZEPAK_RNN\{}_prediction.png'.format(stock), bbox_inches="tight")

