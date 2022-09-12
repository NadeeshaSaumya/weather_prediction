import streamlit as  st
from predict_page import show_predict_page
from explore_page import show_explore_page
from explore_page import load_data

page = st.sidebar.selectbox("Explore or Predict",("Predict","Explore"))
im = st.sidebar.image("picjpg.jpg")
im2 = st.sidebar.image("sun.jpg")
#import predict_page
if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
    load_data()
#predict_page.get_data_columns()