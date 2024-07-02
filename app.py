import streamlit as st
from src.utils import hide_sidebar


st.set_page_config(page_title="Prediction of Commodities",layout="wide",initial_sidebar_state="collapsed")
hide_sidebar()


st.markdown("# Prediction of Prices of Essential Commodities in Karnataka")
st.write("""
        Predicting the prices of essential commodities, 
         such as food grains, vegetables, and other daily necessities, is a critical task for policymakers, traders, and consumers. 
         In the context of Karnataka, a state in India, 
         this involves a combination of data collection, preprocessing, modeling, and validation.""")

b = st.button(label="Click here to know future prices")

if b:
    st.switch_page("pages/predict.py")


    