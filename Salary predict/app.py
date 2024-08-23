import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv("Data//Salary.csv")
x = np.array(data['YearsExperience']).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data['Salary']))

st.title("Salary Predictor") 

nav = st.sidebar.radio("Navigation",["Home", "Prediction", "Contribute"])

if nav == "Home":
    st.image("Data//salary.png", width=500)
if st.checkbox("Show Table"):
    st.table(data)    
    
graph = st.selectbox("What kind of graph",["Non-interactive","Interactive"])

val = st.slider("Filter data using years",0,20)
data = data.loc[data["YearsExperience"]>= val]

if graph == "Non-interactive":
    plt.figure(figsize = (10,5))
    plt.scatter(data["YearsExperience"], data["Salary"])
    plt.ylim(0)
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.tight_layout()
    st.pyplot()

if graph == "Interactive":
    layout = go.Layout(
        xaxis = dict(range=[0,15]),
        yaxis = dict(range=[0,2300000])
    )
    
    fig = go.Figure(data=go.Scatter(x=data["YearsExperience"], y=data["Salary"], mode='markers'),layout=layout)
    st.plotly_chart(fig)

if nav == "Prediction":
    st.header("Knowing your salary")
    val = st.number_input("Enter your experience",0.00,20.00,step=0.25)
    val = np.array(val).reshape(1,-1)
    pred = lr.predict(val)[0]
    
    if st.button("Predict"):
        st.success(f"Your predicted salary is {round(pred)}")
    
if nav == "Contribute":
    st.header("Primary Data contribution")
    ex = st.number_input("Enter your Experience of work",0,20)
    sal = st.number_input("Enter your salary for your work",0,800000,step=10000)
    
    if st.button("submit"):
        to_add = {"YearsExperience": ex, "Salary": sal}
        to_add = pd.DataFrame(to_add, index = [0])
        to_add.to_csv("Data//Salary.csv", mode='a', header=False, index=False)
        
        st.success("Data submitted!")
