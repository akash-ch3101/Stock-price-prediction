symbol= input("Enter Stock Symbol whose upcoming week's stocks you want to predict: ")
import time, datetime
tm =datetime.datetime.today().strftime('%Y-%m-%d')
#all import for the projects
import pandas_datareader.data as wb
web_df1 = wb.DataReader(symbol, 'yahoo', '2015-01-01', tm)
web_df1.to_csv('aapl', sep='\t', encoding='utf-8')
import pandas as pd

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def get_normalised_data(data):
    """
    Normalises the data values using MinMaxScaler from sklearn
    :param data: a DataFrame with columns as  ['index','Open','Close','Volume']
    :return: a DataFrame with normalised value for all the columns except index
    """
    # Initialize a scaler, then apply it to the features
   
    numerical = ['High', 'Low','Open', 'Close', 'Volume']
    data[numerical] = scaler.fit_transform(data[numerical])
    return data

def get_denormalized_data(data):
    scaler.fit(data)
    numerical = ['Close']
    data[numerical] = scaler.inverse_transform(data[numerical])
    return data

def remove_data(data):
    """
    Remove columns from the data
    :param data: a record of all the stock prices with columns as  ['Date','Open','High','Low','Close','Volume']
    :return: a DataFrame with columns as  ['index','Open','Close','Volume']
    """
    # Define columns of data to keep from historical stock data
    item = []
    open = []
    close = []
    volume = []
    high = []
    low = []

    # Loop through the stock data objects backwards and store factors we want to keep
    i_counter = 0
    for i in range(len(data) - 1, -1, -1):
        item.append(i_counter)
        open.append(data['Open'][i])
        close.append(data['Close'][i])
        volume.append(data['Volume'][i])
        high.append(data['High'][i])
        low.append(data['Low'][i])
        i_counter += 1

    # Create a data frame for stock data
    stocks = pd.DataFrame()

    # Add factors to data frame
    stocks['Item'] = item
    stocks['Open'] = open
    stocks['Close'] = pd.to_numeric(close)
    stocks['Volume'] = pd.to_numeric(volume)
    stocks['High']=pd.to_numeric(high)
    stocks['Low'] = pd.to_numeric(low)
    # return new formatted data
    return stocks
stocks = remove_data(web_df1)

#Print the dataframe head and tail
print(stocks.head())
print("---")
print(stocks.tail(1))
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18, 12)


def price(x):
    """
    format the coords message box
    :param x: data to be formatted
    :return: formatted data
    """
    return '$%1.2f' % x


def plot_basic(stocks, title='Google Trading', y_label='Price USD', x_label='Trading Days'):
    """
    Plots basic pyplot
    :param stocks: DataFrame having all the necessary data
    :param title:  Title of the plot 
    :param y_label: yLabel of the plot
    :param x_label: xLabel of the plot
    :return: prints a Pyplot againts items and their closing value
    """
    fig, ax = plt.subplots()
    ax.plot(stocks['Item'], stocks['Close'], '#0A7388')

    ax.format_ydata = price
    ax.set_title(title)

    # Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.show()


def plot_prediction(actual, prediction, train, title='Google Trading vs Prediction', y_label='Price USD', x_label='Trading Days'):
    """
    Plots train, test and prediction
    :param actual: DataFrame containing actual data
    :param prediction: DataFrame containing predicted values
    :param title:  Title of the plot
    :param y_label: yLabel of the plot
    :param x_label: xLabel of the plot
    :return: prints a Pyplot againts items and their closing value
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # Plot actual and predicted close values

    plt.plot(actual, '#00FF00', label='Adjusted Close')
    plt.plot(prediction, '#0000FF', label='Predicted Close')
    plt.plot(train,"#eeefff", label='Training Data')

    # Set title
    ax.set_title(title)
    ax.legend(loc='upper left')

    plt.show()


def plot_lstm_prediction(actual, prediction, title='Google Trading vs Prediction', y_label='Price USD', x_label='Trading Days'):
    """
    Plots train, test and prediction
    :param actual: DataFrame containing actual data
    :param prediction: DataFrame containing predicted values
    :param title:  Title of the plot
    :param y_label: yLabel of the plot
    :param x_label: xLabel of the plot
    :return: prints a Pyplot againts items and their closing value
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # Plot actual and predicted close values

    plt.plot(actual, '#00FF00', label='Adjusted Close')
    plt.plot(prediction, '#0000FF', label='Predicted Close')

    # Set title
    ax.set_title(title)
    ax.legend(loc='upper left')


    plt.show()
plot_basic(stocks)
import numpy as np

stocks = get_normalised_data(stocks)
print(stocks)

print("\n")
print("Open   --- mean :", np.mean(stocks['Open']),  "  \t Std: ", np.std(stocks['Open']),  "  \t Max: ", np.max(stocks['Open']),  "  \t Min: ", np.min(stocks['Open']))
print("Close  --- mean :", np.mean(stocks['Close']), "  \t Std: ", np.std(stocks['Close']), "  \t Max: ", np.max(stocks['Close']), "  \t Min: ", np.min(stocks['Close']))
print("Volume --- mean :", np.mean(stocks['Volume']),"  \t Std: ", np.std(stocks['Volume']),"  \t Max: ", np.max(stocks['Volume']),"  \t Min: ", np.min(stocks['Volume']))
print("High --- mean :", np.mean(stocks['High']),"  \t Std: ", np.std(stocks['High']),"  \t Max: ", np.max(stocks['High']),"  \t Min: ", np.min(stocks['High']))
print("Low --- mean :", np.mean(stocks['Low']),"  \t Std: ", np.std(stocks['Low']),"  \t Max: ", np.max(stocks['Low']),"  \t Min: ", np.min(stocks['Low']))
plot_basic(stocks)
import math
import pandas as pd
import numpy as np

from IPython.display import display

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

import  time #helper libraries
import numpy as np
import math


def scale_range(x, input_range, target_range):
    """

    Rescale a numpy array from input to target range
    :param x: data to scale
    :param input_range: optional input range for data: default 0.0:1.0
    :param target_range: optional target range for data: default 0.0:1.0
    :return: rescaled array, incoming range [min,max]
    """

    range = [np.amin(x), np.amax(x)]
    x_std = (x - input_range[0]) / (1.0*(input_range[1] - input_range[0]))
    x_scaled = x_std * (1.0*(target_range[1] - target_range[0])) + target_range[0]
    return x_scaled, range


    # Divides stock market data intp test and train data set which will generate confusion matrix for the better undestandingdef train_test_split_linear_regression(stocks):
    """
        Split the data set into training and testing feature for Linear Regression Model
        :param stocks: whole data set containing ['Open','Close','Volume'] features
        :return: X_train : training sets of feature
        :return: X_test : test sets of feature
        :return: y_train: training sets of label
        :return: y_test: test sets of label
        :return: label_range: scaled range of label used in predicting price,
    """
    # Create numpy arrays for features and targets
    feature = []
    label = []

    # Convert dataframe columns to numpy arrays for scikit learn
    for index, row in stocks.iterrows():
        # print([np.array(row['Item'])])
        feature.append([(row['Item'])])
        label.append([(row['Close'])])

    # Regularize the feature and target arrays and store min/max of input data for rescaling later
    feature_bounds = [min(feature), max(feature)]
    feature_bounds = [feature_bounds[0][0], feature_bounds[1][0]]
    label_bounds = [min(label), max(label)]
    label_bounds = [label_bounds[0][0], label_bounds[1][0]]

    feature_scaled, feature_range = scale_range(np.array(feature), input_range=feature_bounds, target_range=[-1.0, 1.0])
    label_scaled, label_range = scale_range(np.array(label), input_range=label_bounds, target_range=[-1.0, 1.0])

    # Define Test/Train Split 80/20
    split = .315
    split = int(math.floor(len(stocks['Item']) * split))

    # Set up training and test sets
    X_train = feature_scaled[:-split]
    X_test = feature_scaled[-split:]

    y_train = label_scaled[:-split]
    y_test = label_scaled[-split:]

    return X_train, X_test, y_train, y_test, label_range


def train_test_split_lstm(stocks, prediction_time=1, test_data_size=450, unroll_length=50):
    """
        Split the data set into training and testing feature for Long Short Term Memory Model
        :param stocks: whole data set containing ['Open','Close','Volume'] features
        :param prediction_time: no of days
        :param test_data_size: size of test data to be used
        :param unroll_length: how long a window should be used for train test split
        :return: X_train : training sets of feature
        :return: X_test : test sets of feature
        :return: y_train: training sets of label
        :return: y_test: test sets of label
    """
    # training data
    test_data_cut = test_data_size + unroll_length + 1

    x_train = stocks[0:-prediction_time - test_data_cut].to_numpy()
    y_train = stocks[prediction_time:-test_data_cut]['Close'].to_numpy()

    # test data
    x_test = stocks[0 - test_data_cut:-prediction_time].to_numpy()
    y_test = stocks[prediction_time - test_data_cut:]['Close'].to_numpy()

    return x_train, x_test, y_train, y_test


def unroll(data, sequence_length=24):
    """
    use different windows for testing and training to stop from leak of information in the data
    :param data: data set to be used for unrolling
    :param sequence_length: window length
    :return: data sets with different window.
    """
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)
stocks_data = stocks.drop(['Item'], axis =1)

display(stocks_data.head())
X_train, X_test,y_train, y_test = train_test_split_lstm(stocks_data, 5)

unroll_length = 50
X_train = unroll(X_train, unroll_length)
X_test = unroll(X_test, unroll_length)
y_train = y_train[-X_train.shape[0]:]
y_test = y_test[-X_test.shape[0]:]

print("x_train", X_train.shape)
print("y_train", y_train.shape)
print("x_test", X_test.shape)
print("y_test", y_test.shape)
print(y_train)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


def build_improved_model(input_dim, output_dim, return_sequences):
    """
    Builds an improved Long Short term memory model using keras.layers.recurrent.lstm
    :param input_dim: input dimension of model
    :param output_dim: ouput dimension of model
    :param return_sequences: return sequence for the model
    :return: a 3 layered LSTM model
    """
    model = Sequential()
    model.add(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))

    model.add(Dropout(0.2))

    model.add(LSTM(
        128,
        return_sequences=False))

    model.add(Dropout(0.2))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model


def build_basic_model(input_dim, output_dim, return_sequences):
    """
    Builds a basic lstm model 
    :param input_dim: input dimension of the model
    :param output_dim: output dimension of the model
    :param return_sequences: return sequence of the model
    :return: a basic lstm model with 3 layers.
    """
    model = Sequential()
    model.add(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))

    model.add(LSTM(
        100,
        return_sequences=False))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model
model = build_basic_model(input_dim = X_train.shape[-1],output_dim = unroll_length, return_sequences=True)

# Compile the model
start = time.time()
model.compile(loss='mean_squared_error', optimizer='adam')
print('compilation time : ', time.time() - start)
model.fit(
    X_train,
    y_train,
    epochs=1,
    validation_split=0.05)
predictions = model.predict(X_test)
plot_lstm_prediction(y_test,predictions)
trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))
batch_size = 100
epochs = 10

# build improved lstm model
model = build_improved_model( X_train.shape[-1],output_dim = unroll_length, return_sequences=True)

start = time.time()
#final_model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss='mean_squared_error', optimizer='adam')
print('compilation time : ', time.time() - start)

model.fit(X_train, 
          y_train, 
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_split=0.05
         )
import pandas as pd
pred = pd.DataFrame(predictions)
print(pred)
print(pred)
plot_lstm_prediction(y_test,predictions)
trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))
print(predictions)
import numpy as np
predictions = predictions.reshape(1,446)
predictions = predictions[0][1:]
print(predictions.shape)
predictions = predictions.reshape(89,5)
print(predictions.shape)
predictions = scaler.inverse_transform(predictions)
print(predictions[25][0])



################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################



from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
pd.set_option('expand_frame_repr', False)  
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time 
import yfinance as yahoo_finance
#%matplotlib inline
ticker = symbol
start_time = datetime.datetime(2017, 1, 1)
end_time = datetime.datetime.now().date().isoformat()
connected = False
while not connected:
    try:
        ticker_df = web.get_data_yahoo(ticker, start=start_time, end=end_time)
        connected = True
        print('connected to yahoo')
    except Exception as e:
        print("type error: " + str(e))
        time.sleep( 5 )
        pass   

# use numerical integer index instead of date    
#ticker_df = ticker_df.reset_index()
print(ticker_df.head(5))
df = ticker_df
def computeRSI (data, time_window):
    diff = data.diff(1).dropna()
    # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi
df['RSI'] = computeRSI(df['Adj Close'], 14)
print(df.head())
print(df.tail())
def stochastic(data, k_window, d_window, window):
    
    # input to function is one column from df
    # containing closing price or whatever value we want to extract K and D from
    
    min_val  = data.rolling(window=window, center=False).min()
    max_val = data.rolling(window=window, center=False).max()
    
    stoch = ( (data - min_val) / (max_val - min_val) ) * 100
    
    K = stoch.rolling(window=k_window, center=False).mean() 
    #K = stoch
    
    D = K.rolling(window=d_window, center=False).mean() 


    return K, D
df['K'], df['D'] = stochastic(df['RSI'], 3, 3, 14)
df.head()
df.tail()
def plot_price(df):
    # plot price
    plt.figure(figsize=(18,6))
    plt.plot(df['Adj Close'])
    plt.title('Price chart (Adj Close)')
    plt.show()
    return None

def plot_RSI(df):
    # plot correspondingRSI values and significant levels
    plt.figure(figsize=(18,6))
    plt.title('RSI chart')
    plt.plot(df['RSI'])

    plt.axhline(0, linestyle='--', alpha=0.1)
    plt.axhline(20, linestyle='--', alpha=0.5)
    plt.axhline(30, linestyle='--')

    plt.axhline(70, linestyle='--')
    plt.axhline(80, linestyle='--', alpha=0.5)
    plt.axhline(100, linestyle='--', alpha=0.1)
    plt.show()
    return None

def plot_stoch_RSI(df):
    # plot corresponding Stoch RSI values and significant levels
    plt.figure(figsize=(18,6))
    plt.title('stochRSI chart')
    plt.plot(df['K'])
    plt.plot(df['D'])

    plt.axhline(0, linestyle='--', alpha=0.1)
    plt.axhline(20, linestyle='--', alpha=0.5)
    #plt.axhline(30, linestyle='--')

    #plt.axhline(70, linestyle='--')
    plt.axhline(80, linestyle='--', alpha=0.5)
    plt.axhline(100, linestyle='--', alpha=0.1)
    plt.show()
    return None

def plot_all(df):
    plot_price(df)
    plot_RSI(df)
    plot_stoch_RSI(df)
    return None
plot_all(df)
# weekly timeframe aggregation

agg_dict = {'Open': 'first',
          'High': 'max',
          'Low': 'min',
          'Close': 'last',
          'Adj Close': 'last',
          'Volume': 'mean'}

# resampled dataframe
# 'W' means weekly aggregation
df_res = df.resample('W').agg(agg_dict)
df_res['RSI'] = computeRSI(df_res['Adj Close'], 14)
df_res['K'], df_res['D'] = stochastic(df_res['RSI'], 3, 3, 14)
def plot_weekly_stoch_RSI(df_res):
    # WEEKLY plot corresponding Stoch RSI values and significant levels
    plt.figure(figsize=(18,6))
    plt.title('WEEKLY stochRSI chart')

    df_res = df_res.reset_index()
    plt.plot(df_res['Date'], df_res['K'].fillna(0))
    # NaN to zeros so plot is in scale
    plt.plot(df_res['Date'], df_res['D'].fillna(0))
    # NaN to zeros so plot is in scale

    #df_res.reset_index().plot(x='Date', y='K', figsize=(15,5))
    #df_res.reset_index().plot(x='Date', y='D', figsize=(15,5))

    plt.axhline(0, linestyle='--', alpha=0.1)
    plt.axhline(20, linestyle='--', alpha=0.5)
    #plt.axhline(30, linestyle='--')

    #plt.axhline(70, linestyle='--')
    plt.axhline(80, linestyle='--', alpha=0.5)
    plt.axhline(100, linestyle='--', alpha=0.1)
    plt.show()
    return None

def plot_mixed(df, df_res):
    plot_price(df)
    plot_RSI(df)    
    plot_weekly_stoch_RSI(df_res)
    return None
plot_mixed(df, df_res)

