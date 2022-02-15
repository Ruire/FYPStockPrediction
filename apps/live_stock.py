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
    col1,col2,col3 = st.columns(3)
    with col1:
        st.write("Number of Full Time Employees: ")

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
