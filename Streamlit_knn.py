#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import streamlit as st
from pickle import dump
from pickle import load



st.title('Model Deployment: KNN Classifier')



st.sidebar.header('User Input Parameters')



def user_input_features():
    industrial_risk= st.sidebar.selectbox('Industrial risk',('0','0.5','1'))
    management_risk= st.sidebar.selectbox('Management risk',('0','0.5','1'))
    financial_flexibility= st.sidebar.selectbox('Financial flexibility',('0','0.5','1'))
    credibility= st.sidebar.selectbox('Credibility',('0','0.5','1'))
    competitiveness= st.sidebar.selectbox('Competitiveness',('0','0.5','1'))
    operating_risk= st.sidebar.selectbox('Operating risk',('0','0.5','1'))
    data = {'industrial_risk':industrial_risk,
            ' management_risk':management_risk,
            ' financial_flexibility':financial_flexibility,
            ' credibility':credibility,
            ' competitiveness':competitiveness,
            ' operating_risk':operating_risk}
    features = pd.DataFrame(data,index = [0])
    return features

df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)

#load the model from disk
loaded_model= load(open('finalized_model.sav', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Non Bankruptcy' if prediction_proba[0][1] > 0.5 else 'Bankruptcy')

st.subheader('prediction probability ')
st.write(prediction_proba)

