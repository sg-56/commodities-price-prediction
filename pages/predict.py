import streamlit as st
from datetime import datetime,timedelta
from src.utils import hide_sidebar,GetCommodities
from src.pipelines.predict_pipeline import PredictPipeline
from src.pipelines.predict_pipeline import CustomData
st.set_page_config(page_title="Prediction Page",layout="wide")
 
    
hide_sidebar()



cols = st.columns([1,2,1])
with cols[1]:
    st.markdown("# Predict Prices of commodities")
    with st.form(key='Input_form',border=True):
        commodity = st.selectbox(label="Select Commodity : ",options=GetCommodities())
        market = st.radio(label="Select Market Type: ",options=['Retail','Wholesale'])
        date = st.date_input(label="Enter the Date : ",value=datetime.now(),min_value=datetime.now(),max_value=datetime.today()+timedelta(days = 1000),format="DD/MM/YYYY")
        submit = st.form_submit_button(label="Get Price")

with st.container(border=True):
    with cols[1]:
        if submit:
            df = CustomData(date = date,market=market,commodity=commodity).get_data_as_data_frame()
            #st.write(df)
            pipe = PredictPipeline()
            a = pipe.predict(df)
            st.write(f"The price of {commodity} on {date} will be approximately : ")
            a = round(a[0],3)
            st.markdown(f"## Rs {str(a)}/-")
