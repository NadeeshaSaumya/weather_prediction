from tkinter import Image
from predict_page import load_model, data, __model
import streamlit as st
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder



@st.cache
def load_data():
    df = pd.read_csv('seattle-weather.csv')
    df1 = df.drop(['date'], axis=1)
    #df1.iloc[:, :].boxplot()
    Q1 = df1.quantile(0.25)
    Q3 = df1.quantile(0.75)
    IQR = Q3 - Q1
    Out_row = ((df1 < (Q1 - 1.5 * IQR)) | (df1 > (Q3 + 1.5 * IQR))).any(axis=1)
    df2 = df1[~Out_row]
    x = df2.drop(['weather'], axis=1)
    y = df2['weather']
    smt = SMOTE()
    x_sm, y_sm = smt.fit_resample(x, y)
    df3 = pd.concat([x_sm,y_sm], axis=1)
    return df3

df3 = load_data()

@st.cache
def abc():
    x_sm = df3.drop(['weather'],axis = 1)
    y_sm = df3["weather"]
    x_train, x_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size=0.3)
    wea = RandomForestClassifier(n_estimators=150, criterion='entropy')
    wea.fit(x_train, y_train)
    accu = wea.score(x_test, y_test)
    return accu
accu = abc()


def show_explore_page():
    st.title("Exploration of weather prediction dataset")
   # image = Image.open('C:/Users/Nadeesha Saumya/ML/train1.jpg')
   # st.image(image, caption='ML', use_column_width=True)
    st.subheader('Data Infirmation: ')
    st.dataframe(df3)
    st.subheader('Statistics of the data: ')
    st.write(df3.describe())

    st.subheader('Bar graph of y value counts: ')
    data = df3["weather"].value_counts()
    st.bar_chart(data)

    st.subheader('Boxplot of weather vs maximum temperature: ')
    fig, ax = plt.subplots()
    (sns.boxplot(df3["weather"],df3["temp_max"],data = df3))
    st.pyplot(fig)

    st.subheader('Boxplot of weather vs minimum temperature: ')
    fig, ax = plt.subplots()
    (sns.boxplot(df3["weather"], df3["temp_min"], data=df3))
    st.pyplot(fig)

    st.subheader('Boxplot of weather vs wind: ')
    fig, ax = plt.subplots()
    (sns.boxplot(df3["weather"], df3["wind"], data=df3))
    st.pyplot(fig)

    st.subheader('Boxplot of weather vs precipitation: ')
    fig, ax = plt.subplots()
    (sns.boxplot(df3["weather"], df3["precipitation"], data=df3))
    st.pyplot(fig)

    st.subheader('Model Test Accuracy Score:')
    st.write(f"accuracy_score of the model is {accu}")


'''   
    #st.metrics.confusion_matrix(test,pred)

    x_sm = df3.drop(['weather'], axis=1)
    y_sm = df3["weather"]
    x_train, x_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size=0.3)
    wea = RandomForestClassifier(n_estimators=150, criterion='entropy')
    wea.fit(x_train, y_train)
  #  test = y_test
    pred = wea.predict(x_test)
    st.subheader("Confusion Matrix: ")
    le_te = LabelEncoder()
    test = le_te.fit_transform(y_test)
    le_pe = LabelEncoder()
    predi = le_pe.fit_transform(pred)


    plot_confusion_matrix([[wea,test, predi]])

   # cm = metrics.confusion_matrix(test,pred)


    st.subheader("HeatMap: ")'''