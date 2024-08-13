import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from src.utils import GetCommodities,getinfo,getImages
import plotly.express as px
import streamlit_shadcn_ui as ui

st.set_page_config(page_title="Data Visualisations",layout="wide")


color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

data = pd.read_csv('notebooks/Dataset.csv',parse_dates=True)
data['date'] = pd.to_datetime(data['date'])

st.markdown("# Historical Price Data ")

with st.container(border=True):
    commodity = st.selectbox(label="Select A Commodity",options=GetCommodities())
    info_cols = st.columns(2)
    with info_cols[0]:
        ui.metric_card(title="Commodity Info",content=getinfo(commodity_name=commodity))
    with info_cols[1]:
        st.image(getImages(commodity_name=commodity))
    f = data.loc[(data["commodity"] == commodity)]
    metrics = st.columns(3)
    with metrics[0]:
        min_retail_price = f.loc[f['market'] == "Retail"]['price'].min()
        max_retail_price = f.loc[f['market'] == "Retail"]['price'].max()
        ui.metric_card(title="Highest Retail Value",content=max_retail_price)
    with metrics[1]:
        ui.metric_card(title="Lowest Retail Value ",content=min_retail_price)
    fig = px.line(data_frame= f,x='date',y = ['price'],color="market")
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
        )
    )
    st.plotly_chart(fig)
    cols = st.columns(2)
    with cols[0]:
        fig = px.density_contour(data_frame=f,x='date',y = 'price',title="Distribution of price values",color='market')
        st.plotly_chart(fig)
    with cols[1]:
        d = f.groupby(by=['date','market'])['price'].mean().reset_index()
        d['date'] = pd.to_datetime(d['date']).dt.year
        
        fig = px.box(data_frame=d,x='date',y='price',title="Average price by year",color='market')
        st.plotly_chart(fig)
        # st.write(d)
    
        
