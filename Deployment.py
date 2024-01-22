#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import streamlit as st 
from xgboost import XGBRegressor
from pickle import dump
from pickle import load
from pandas import DataFrame


# In[16]:


st.title('Model Deployment: XGBoost Regression')


# In[17]:


df = pd.read_csv("C:/Users/preet/OneDrive/Desktop/ExcelR/Data Science/Project/temperature_data.csv")
df.drop(["motor_speed"],inplace=True,axis = 1)
df.drop(["profile_id"],inplace=True,axis = 1)
df = df.dropna().reset_index()
df.drop(["index"],inplace=True,axis = 1)


# In[18]:


st.sidebar.header('User Input Parameters')


# In[19]:


def user_input_features():
    ambient = st.sidebar.number_input("Insert Ambient", format="%.7f")
    coolant = st.sidebar.number_input("Insert Coolant", format="%.7f")
    stator_tooth = st.sidebar.number_input("Insert Stator tooth", format="%.7f")
    stator_winding = st.sidebar.number_input("Insert Stator winding", format="%.7f")
    stator_yoke = st.sidebar.number_input("Insert Stator Yoke", format="%.7f")
    i_d = st.sidebar.number_input("Insert I_d", format="%.7f")
    pm = st.sidebar.number_input("Insert pm ", format="%.7f")
    u_q = st.sidebar.number_input("Insert u_q", format="%.7f")
    i_q = st.sidebar.number_input("Insert I_q", format="%.7f")
    torque = st.sidebar.number_input("Insert Torque", format="%.7f")
    u_d = st.sidebar.number_input("Insert u_d", format="%.7f")
    profile_id = st.sidebar.number_input("Insert Profile Id", format="%.7f")
    data = {'ambient':ambient,
            'coolant':coolant,
            'stator_tooth':stator_tooth,
            'stator_winding':stator_winding,
            'stator_yoke':stator_yoke,
            'i_d':i_d,
            'pm':pm,
            'u_q':u_q,
            'i_q':i_q,
            'torque':torque,
            'u_d':u_d,
            'profile_id':profile_id
            }
    features = pd.DataFrame(data,index = [0])
    return features 


# In[20]:


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# In[21]:


# load the model from disk
loaded_model = load(open("C:/Users/preet/OneDrive/Desktop/ExcelR/Data Science/Project/filename", 'rb'))


# In[22]:


cols_when_model_builds = loaded_model.get_booster().feature_names


# In[23]:


df = df[cols_when_model_builds]


# In[24]:


prediction = loaded_model.predict(df)


# In[25]:


st.subheader('Motor Speed Prediction')
st.write(prediction)


# In[26]:


output=pd.concat([df,pd.DataFrame(prediction)],axis=1)


# In[27]:


output.to_csv('output.csv')


# In[ ]:




