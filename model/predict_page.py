import streamlit as st
import pickle
import numpy as np
__model = None

def load_model():
    global __model
    if __model is None:
        with open('weather_prediction_model.pickle', 'rb') as f:
            __model = pickle.load(f)
        print("loading saved artifacts...done")
       # return __model


data = load_model()
try:
   regressor = data[int(float("model"))]
except ValueError:
   pass



def show_predict_page():
    st.title("Predict Weather Condition")

    st.write("""### We need some information to predict the weather""")

    precipitation = st.slider("Precipitation",0.0,10.0,0.8)
    temp_max = st.slider("Maximum Temperature",-5.0,40.0,17.0)
    temp_min = st.slider("Minimum Temperature",-10.0,20.0,8.5)
    wind = st.slider("Wind",0.0,10.0,3.0)

    ok = st.button("Predict Weather")
    if ok:
        x = np.array([[precipitation,temp_max,temp_min,wind]])
        x =x.astype(float)
        a = __model.predict(x)
        st.subheader(f"predicted weather condition is {a[0]}!")