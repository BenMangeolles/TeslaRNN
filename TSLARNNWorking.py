import yfinance as yf
import pandas as pd
from collections import deque
import random
import numpy as np
import time
from sklearn import preprocessing
import csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
#These are all of various modules/packages used if you want to try run it you need to pip install all of them

TSLADF = 'TeslaRNN\\YFTSLA.csv' #Location of the data that goes into the data frame define below, this includes price and volume of the TSLA stock


#Global Vars called at end of code: 
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
EPOCHS = 10
BATCH_SIZE = 8
NAME = "RNN_FIRST_TSLA_TRY"


#NOTE: Look up what EPOCH's and Batch sizes are, very important in not overfitting a RNN


#SEQ_LEN varaible is the amount of data 'labels' that will be input at once in RNN. 60 means every hour as the data is in minutes

#Skip to line 82 and then back to here
def classify(current, future): #Rule for the targets of the AI
    if float(current) > float(future):
        return 0
    else:
        return 1

def preprocess_df(df):
    df = df.drop("future", 1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            #inplace means fill all error/ 0 values with a null value rather than outputting an error
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    #deque is useful when using lists, 

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    return np.array(X), y

#Most important part of NN is formatting the data so that it is all compatible to be sorted through so you use a data DataFrame
main_df = pd.DataFrame()
#Defined the variable main_df as a data frame, this uses the Pandas packages which I have renamed pd
df = pd.read_csv(TSLADF, names=["Datetime","Open","High","Low","Close","Volume","Dividends","Stock Splits"])
#Reads the file with the data in and gives each column of data a heading
df['Datetime'] = pd.to_datetime(df['Datetime'], format="%Y-%m-%d %H:%M")
#The formatting of time the data is from is strange so I had to reformat it
#This is how you format time so computer understands it in python and java
df.set_index("Datetime", inplace=True)
#This sets the 'index' of the data frame which is what makes it a reccurrent neural network, this is unique to RNN and can be thought of as time Series
df = df[["Close","Volume"]]
#Isolates the columns that I want to use the data from and removes the rest from the data frame
main_df = df
#Make the main_df variable defined earlier have the properties of the df that I was creating
main_df["future"] = main_df["Close"].shift(-FUTURE_PERIOD_PREDICT)
#Used for training, imagine two identical columns of the closing price (the price that matters) of a stock next to each other
#This shifts one of the columns down by a specific time period, in this instance by 3 rows as deined at the start of code
#This then creates values that the RNN can use as a 'target' as it can compare its predicition of the stock price in 3 minutes to the actual stock price 3 minutes later which is defined in this new column 'future'
main_df["target"] = list(map(classify, main_df["Close"], main_df["future"]))


times = sorted(main_df.index.values)
last5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last5pct)]
main_df = main_df[(main_df.index < last5pct)]
train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

#TLDR factors/parameters/vars that matter LSTM, optimiser, Adam, Dropout
model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='sigmoid'))
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
             optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
filepath = 'RNN_Final-{epoch:02d}'
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validation_x, validation_y), callbacks=[tensorboard, checkpoint])

model.save()
