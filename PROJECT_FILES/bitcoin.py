# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 21:35:30 2023

@author: nihit
"""

import streamlit as st

def main():
    st.set_page_config(page_title="Time Series Analysis for Bitcoin Price Prediction")

    st.markdown(
        """
        <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-image: url("bitcoin1.jpg");
            background-size: cover;
            background-repeat: no-repeat;
        }
        
        main {
            margin: 100px;
            padding: 20px;
            bottom: 20px;
            margin-top: 0px;
            margin-bottom: 100px;
            padding-bottom: 10px;
            font-size: 20px;
        }
        
        img.background {
            filter: blur(5px);
        }
        
        header {
            background-color: whitesmoke;
            color: black;
            padding: 10px;
            text-align: center;
        }
        
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: whitesmoke;
            color: black;
            text-align: center;
            text-decoration-color: black;
        }
        
        .title {
            color: whitesmoke;
        }
        
        h2 {
            color: whitesmoke;
        }
        
        header h1,
        footer p {
            margin: 0;
            color: black;
            background-color: whitesmoke;
        }
        
        main {
            padding: 20px;
        }
        
        h1 {
            color: whitesmoke;
            background-color: black;
        }
        
        .prediction-link {
            display: inline-block;
            padding: 10px 20px;
            background-color: whitesmoke;
            color: black;
            text-decoration: none;
            margin-top: 30px;
            margin: 70px 70px;
        }
        
        .predict {
            padding-bottom: 10px;
            padding-left: 20px;
            padding-right: 20px;
            color: white;
        }
        
        .submit-button {
            padding: 10px 20px;
            background-color: whitesmoke;
            color: black;
            border: none;
            cursor: pointer;
        }
        
        #prediction-result {
            margin-top: 20px;
        }
        
        p {
            color: whitesmoke;
        }
        
        footer {
            color: black;
            font-size: 20px;
            margin: 5px;
            font-weight: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <header>
            <h1>Time Series Analysis for Bitcoin Price Prediction</h1>
        </header>
        <div class="blur-background">
            <main>
                <p>
                    Bitcoin is known as a type of cryptocurrency because it uses cryptography to keep it secure.
                    There are no physical bitcoins, only balances kept on a public ledger that everyone has transparent
                    access to (although each record is encrypted). All Bitcoin transactions are verified by a massive
                    amount of computing power via a process known as "mining." Bitcoin is not issued or backed by any
                    banks or governments, nor is an individual bitcoin valuable as a commodity.
                </p>
                <p>
                    Bitcoin is one of the first digital currencies to use peer-to-peer (P2P) technology to facilitate
                    instant payments. The independent individuals and companies who own the governing computing power and
                    participate in the Bitcoin network-Bitcoin "miners"—are in charge of processing the transactions on
                    the blockchain and are motivated by rewards (the release of new Bitcoin) and transaction fees paid in
                    Bitcoin
                </p>
                <p>
                    To get started, proceed to the prediction page. By clicking on Go to Prediction.
                </p>
                
            </main>
        </div>
        <div class="footer">
            <footer>
                <p>&copy; 2023 AI Project. All rights reserved.</p>
            </footer>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()


import yfinance as yf
from prophet import Prophet

df = yf.download('BTC-USD')
df.reset_index(inplace=True)
df = df[['Date','Adj Close']]
df.columns = ['ds','y']

model = Prophet()
model.fit(df)

future_dates = model.make_future_dataframe(periods=100,freq='D')
prediction = model.predict(future_dates)

selected_date_str = st.text_input("Please enter a date (YYYY-MM-DD)")
if selected_date_str in prediction['ds'].astype(str).values:
    price_prediction = prediction.loc[prediction['ds'].astype(str) == selected_date_str, 'yhat'].values[0]
    st.write("Bitcoin price on", selected_date_str, "is", round(price_prediction, 2), "cents")
else:
    st.write("Invalid date")