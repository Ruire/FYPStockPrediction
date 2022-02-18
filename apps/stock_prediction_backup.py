import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st #
import pandas as pd #
import yfinance as yf #
import datetime
from datetime import date
from datetime import datetime
from datetime import timedelta
import time
import requests
import io
import math
import matplotlib.pyplot as plt
import numpy as np
import plotly
import cufflinks as cf
import warnings
warnings.filterwarnings('ignore')

from bs4 import BeautifulSoup #req.text
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#ML
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

def prediction_test():

    df = yf.download(user_input_gbl, start = date(2021,1,24), end = date.today(), progress = False )
    chart_data = pd.DataFrame(
         df[option_gbl])

    #count the number of entries
    entry_count = len(df[option_gbl])
    training_set = df[option_gbl]       #create a training set using the dataframe retrieved
    training_set = pd.DataFrame(training_set)   #converting variable into DataFrame
    df.isna().any()
    sc = MinMaxScaler(feature_range = (0, 1))         #
    training_set_scaled = sc.fit_transform(training_set)        #for model fitting to ensure that training data and test error data are minimal
    X_train = []
    y_train = []

    for i in range(60, entry_count):  #60 because take data from day 1 to day 60, then making predicition on 61st day. #
        X_train.append(training_set_scaled[i-60:i, 0])      #appending into the training sets with model fit applied
        y_train.append(training_set_scaled[i, 0])           #appending into the training sets with model fit applied
    X_train, y_train = np.asarray(X_train).astype(np.float32), np.asarray(y_train).astype(np.float32)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    #Training model
    model = Sequential()
    #Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units = 1))
    model.compile(optimizer = "adam", loss = "mean_squared_error") #adding adam optimizer( can refer to proposal to see how it works)

    # Fitting the RNN to the Training set, epoch is the number of times the data is passed through the model
    model.fit(X_train, y_train, epochs = 1, batch_size = 32)

    price_data = df[option_gbl]
    price_data.fillna(value=0, inplace=True)

    dataset_total = pd.concat((df[option_gbl], price_data), axis = 0)       #compiling and reshaping the data to become readable in a list format, while ensuring that the data is ordered
    inputs = dataset_total[len(dataset_total) - len(price_data) - 60:].values   #compiling and reshaping the data to become readable in a list format, while ensuring that the data is ordered
    inputs = inputs.reshape(-1,1)   #compiling and reshaping the data to become readable in a list format, while ensuring that the data is ordered
    inputs = sc.transform(inputs)   #compiling and reshaping the data to become readable in a list format, while ensuring that the data is ordered
    X_test = []
    for i in range(60, len(df) + 67):   #compiling and reshaping the data to become readable in a list format, while ensuring that the data is ordered
        X_test.append(np.array(inputs[i-60:i, 0]).tolist())     #compiling and reshaping the data to become readable in a list format, while ensuring that the data is ordered
    # X_test_pred = np.asarray(X_test).astype(np.float32)
    X_test_len = max(map(len, X_test))      #compiling and reshaping the data to become readable in a list format, while ensuring that the data is ordered
    X_test_pred = np.array([xi+[0.0]*(X_test_len-len(xi)) for xi in X_test]).astype(np.float32)     #compiling and reshaping the data to become readable in a list format, while ensuring that the data is ordered

    # X_test = np.reshape(X_test_pred, (X_test_pred.shape[0], X_test_pred.shape[1]))
    X_test = X_test_pred                        #line 98-104 placing the predicted data values into a model and then coverting it into a dataframe
    stock_dates = df.index
    real_stock_price = df.iloc[:,0:4]
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    # st.text(predicted_stock_price)
    predicted_stock_price = pd.DataFrame(predicted_stock_price,columns = ['Predicted Stock Price'])

    global var_real_stock_price
    global var_predicted_stock_price
    global var_stock_dates
    var_real_stock_price = real_stock_price
    var_predicted_stock_price = predicted_stock_price
    var_stock_dates = stock_dates

def news_sentiment(user_input):
    n = 5
    tickers = [user_input]

    finwiz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    for ticker in tickers:
        url = finwiz_url + ticker
        req = Request(url=url, headers={'user-agent': 'Mozilla/5.0'})
        resp = urlopen(req)
        html = BeautifulSoup(resp, features="lxml")
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    try:
        for ticker in tickers:
            df = news_tables[ticker]
            df_tr = df.findAll('tr')

            st.write('Recent News Headlines for {}: '.format(ticker))

            for i, table_row in enumerate(df_tr):
                a_text = table_row.a.text
                td_text = table_row.td.text
                td_text = td_text.strip()
                st.write(a_text, '(', td_text, ')')
                if i == n - 1:
                    break
    except KeyError:
        pass
    # Iterate through the news
    parsed_news = []
    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text()
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]

            else:
                date = date_scrape[0]
                time = date_scrape[1]

            ticker = filename.split('')[0]

            parsed_news.append([ticker, date, time, text])

    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()

    columns = ['Ticker', 'Date', 'Time', 'Headline']
    news = pd.DataFrame(parsed_news, columns=columns)
    scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

    df_scores = pd.DataFrame(scores)
    news = news.join(df_scores, rsuffix='_right')

    # View Data
    news['Date'] = pd.to_datetime(news.Date).dt.date

    unique_ticker = news['Ticker'].unique().tolist()
    news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

    values = []
    for ticker in tickers:
        dataframe = news_dict[ticker]
        dataframe = dataframe.set_index('Ticker')
        dataframe = dataframe.drop(columns=['Headline'])
        st.write(dataframe.head())

        mean = round(dataframe['compound'].mean(), 2)
        values.append(mean)

    df = pd.DataFrame(list(zip(tickers, values)), columns=['Ticker', 'Mean Sentiment'])
    df = df.set_index('Ticker')
    df = df.sort_values('Mean Sentiment', ascending=False)
    st.write(df)

def app():
    global user_input_gbl
    global start_date_gbl
    global option_gbl
    st.title('View Stock Prediction')
    with st.expander("Description"): #description for stock prediction
        st.write("Enter a stock symbol of your choice in the text field below (ETF/Bonds/Commodities). Anything that is in Yahoo Finance will be shown here.")
        st.write("Run the prediction by pressing the run prediction button. Select the data type to perform the prediction with (Open/High/Low/Close).")
    col1, col2, = st.columns(2)         #creating 2 columns
    with col1:
        user_input = st.text_input("ENTER STOCK SYMBOL")        #text input for entering stock symbol
        user_input_gbl = user_input
        if user_input != '':                                    #if user is not null, obtain the first date of the stock recorded
            df = yf.download(user_input)                        #if user is not null, obtain the first date of the stock recorded
            date_index = str(df.index[0])                       #if user is not null, obtain the first date of the stock recorded
            dt_obj = datetime.strptime(date_index, "%Y-%m-%d %H:%M:%S")     #if user is not null, obtain the first date of the stock recorded
            start_date_gbl = dt_obj                                         #if user is not null, obtain the first date of the stock recorded
    with col2:
        option = st.selectbox('Select OHLC',('Open','High','Low','Close'))  #select the different types of stock data for the prediction
        option_gbl = option
            #select date range if you want, or you can just see current time stock
    if st.button("Run Prediction"):
        prediction_test()               #run the above mentioned method when the button is pressed
        news_sentiment(user_input_gbl)
        data_predicted_stock_price = var_predicted_stock_price["Predicted Stock Price"].to_numpy()  #converting variable to numpy variable
        date_index = (list(var_real_stock_price.index))             #creating an index with dates
        for i in range(0, len(data_predicted_stock_price) - len(date_index)):           #appending the dates for the intended predicted values
            pred_date = date_index[len(date_index)-1]                                   #appending the dates for the intended predicted values
            pred_date += timedelta(days=1)                                              #appending the dates for the intended predicted values
            date_index.append(pred_date)                                                #appending the dates for the intended predicted values

        data_real_stock_price = var_real_stock_price[option_gbl].to_numpy().tolist()      #   #appending the dates for the intended predicted values
        for i in range(0, len(data_predicted_stock_price) - len(data_real_stock_price)):        #appending the dates for the intended predicted values
            data_real_stock_price.append(np.nan)            #appending the dates for the intended predicted values

        df = pd.DataFrame(
            {
                "Predicted Stock Price": data_predicted_stock_price,        #placing the predicted stock price values into DataFrame
                "Real Stock Price": data_real_stock_price,                  # placing real stock price values into dataframe
            },
            index=date_index                                            #modifying date index to be the index
        )

        chart = df
        st.line_chart(chart)            #plot chart with predicted stock price and real stock price
        with st.container():
            latest_predicted_price = var_predicted_stock_price._get_value(len(var_predicted_stock_price)-1,'Predicted Stock Price') #find the values of the predicted values, they are 7 values
            latest_real_price = data_real_stock_price[len(data_real_stock_price)-8]
            price_percentage = latest_predicted_price-latest_real_price/100     #method of seeing if should hold/buy/sell from line 161 to line 175
            st.title(f"7 day forecast for {user_input} ")
            st.write(df["Predicted Stock Price"].tail(7))

            if (price_percentage > 10):
                coa = "Buy large amounts"
            elif(price_percentage < 10 and price_percentage > 5):
                coa = "Buy small amounts"
            elif(price_percentage < 5 and price_percentage > -5):
                coa = "Hold"
            elif(price_percentage < -5 and price_percentage > -10):
                coa = "Sell small amounts"
            else:
                coa = "Sell large amounts"
            st.write(" Prediction's recommended course of action: " + coa)

            recommendations = pd.DataFrame(yf.Ticker(user_input).recommendations)
            st.header("Analyst Recommendations from Yahoo Finance")
            st.write(recommendations.tail())
