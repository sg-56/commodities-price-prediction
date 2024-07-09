import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import GetCommodities

st.set_page_config(page_title="Data Visualisations",layout="wide")


color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

data = pd.read_csv('notebooks/Dataset.csv',parse_dates=True)

st.markdown("# Historical Price Data ")

with st.container():
    commodity = st.selectbox(label="Select A Commodity",options=GetCommodities())
    f = data.loc[(data["commodity"] == commodity)]
    st.line_chart(data = f,x = 'date',y='price',color='market')