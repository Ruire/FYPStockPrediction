import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup as soup
import urllib
from urllib.request import Request, urlopen


def page_settings():
    with st.container():
        components.html(
        """
        <!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div id="tradingview_d0726"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget(
  {
  "width": 800,
  "height": 700,
  "symbol": "NASDAQ:AAPL",
  "interval": "D",
  "timezone": "Etc/UTC",
  "theme": "light",
  "style": "1",
  "locale": "en",
  "toolbar_bg": "#f1f3f6",
  "enable_publishing": false,
  "allow_symbol_change": true,
  "container_id": "tradingview_d0726"
}
  );
  </script>
</div>
<!-- TradingView Widget END -->
        """,height = 700
        )

    st.title('View fundamentals of stock')
    user_input = st.text_input("ENTER STOCK SYMBOL",value="AAPL")
    information = []
    information.append(yf.Ticker(user_input).info)
    df = pd.DataFrame(information)
    st.write(df)
    col1,col2,col3 = st.columns(3)
    fte = str(df.iloc[0,2])
    industry = str(df.iloc[0,12])
    profit_margins = str(df.iloc[0,14])
    gross_margins = str(df.iloc[0,15])
    revenue_growth = str(df.iloc[0,17])
    operating_margins = str(df.iloc[0,18])
    return_on_assets = str(df.iloc[0,28])
    target_mean_price = str(df.iloc[0,30])
    target_high_price = str(df.iloc[0,33])
    total_cash_per_share = str(df.iloc[0,37])
    financial_currency = str(df.iloc[0,38])
    long_name = str(df.iloc[0,44])
    market = str(df.iloc[0,52])
    forward_eps = str(df.iloc[0,59])
    last_fiscal_year_end = str(df.iloc[0,69])
    earnings_quarterly_growth = str(df.iloc[0,92])
    peg_ratio = str(df.iloc[0,95])
    payout_ratio = str(df.iloc[0,108])
    average_daily_volume_10_d = str(df.iloc[0,112])
    fifty_day_avg = str(df.iloc[0,114])
    market_cap = str(df.iloc[0,132])
    average_volume = str(df.iloc[0,135])
    pre_market_price = str(df.iloc[0,150])
    target_median_price = str(df.iloc[0,24])
    target_low_price = str(df.iloc[0,20])
    gross_profits = str(df.iloc[0,22])
    free_cash_flow = str(df.iloc[0,23])
    with col1:
        st.write("Company Name: " + long_name)
        st.write("Financial Currency: " + financial_currency)
        st.write("Market: " + market)
        st.write("Industry: " + industry)
        st.write("Market Cap: " + market_cap)
        st.write("Number of Full Time Employees: " + fte)
        st.write("Gross Profits: " + gross_profits)
        st.write("Last Fiscal Year : " + last_fiscal_year_end)
        st.write("Quarterly Earnings Growth: " + earnings_quarterly_growth)


    with col2:

        st.write("Payout Ratio (%): " + payout_ratio)
        st.write("Profit Margins (%): " + profit_margins)
        st.write("Gross Margins (%): " + gross_margins)
        st.write("Revenue Growth (%): " + revenue_growth)
        st.write("Operating Margins (%): " + operating_margins)
        st.write("Return on Assets (%): " + return_on_assets)
        st.write("Total Cash Per Share: " + total_cash_per_share)
        st.write("Average Daily Volume 10 days: " + average_daily_volume_10_d)
        st.write("Fifty Day Average: " + fifty_day_avg)


    with col3:
        st.write("Average Trade Volume: " + average_volume)
        st.write("Free Cash Flow: " + free_cash_flow)
        st.write("Pre Market Price: " + pre_market_price)
        st.write("Target Low Price: " + target_low_price)
        st.write("Target Median Price: " + target_median_price)
        st.write("Target Mean Price: " + target_mean_price)
        st.write("Target High Price: " + target_high_price)
        st.write("Peg Ratio (%): " + peg_ratio)
        st.write("Forward Eps: " + forward_eps)

        #news scraper
    st.title("Latest News column")
    url = ("http://finviz.com/quote.ashx?t=" + user_input.lower())
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    html = soup(webpage, "html.parser")

    def get_news():
        try:   # Find news table
            news = pd.read_html(str(html), attrs={'class': 'fullview-news-outer'})[0]
            links = []
            for a in html.find_all('a', class_="tab-link-news"):
                links.append(a['href'])
                        # Clean up news dataframe
            news.columns = ['Date', 'News Headline']
            news['Article Link'] = links
            news = news.set_index('Date')
            return news

        except Exception as e:
            return e
    st.write(get_news())

def app():
    with st.container():
        page_settings()
