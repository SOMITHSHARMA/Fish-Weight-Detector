#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")


# In[3]:


from IPython import get_ipython


# In[4]:


fish_data = pd.read_csv('fish.csv')


# In[5]:


fish_data


# In[6]:


fish_data.head()


# In[7]:


fish_data.tail()


# In[8]:


fish_data.shape


# In[9]:


fish_data.columns


# In[10]:


fish_data.isnull().sum()


# In[11]:


fish_data.duplicated().sum()


# In[12]:


fish_data.describe()


# In[13]:


fish_data.info()


# In[14]:


fish_data.nunique()


# In[15]:


fish_data['Species'].unique()


# In[16]:


plt.figure(figsize=(15,6))
sns.countplot('Species' , data = fish_data, palette = 'hls')
plt.xticks(rotation=90)
plt.show()


# In[17]:


pip install plotly


# In[18]:


import plotly.express as px


# In[19]:


fig = px.histogram(fish_data , x = 'Species', color = 'Species')
fig.show()


# #                                                 pair plot

# In[20]:


sns.pairplot(fish_data)


# #                finding the correlation to find out how the attributes are         
# #                                      dependent on each other

# In[21]:


fish_data.corr()


# In[22]:


plt.figure(figsize=(15,6))
sns.heatmap(fish_data.corr(),annot=True)
plt.show()


# In[23]:


# Box Plot
plt.figure(figsize=(15,6))
sns.boxplot(fish_data['Weight'])
plt.xticks(rotation = 90)
plt.show()


# #                                       Removing Outliers
# #                                      (lying outside graph)

# In[24]:


fish_weight = fish_data['Weight']
Q3 = fish_weight.quantile(0.75)
Q1 = fish_weight.quantile(0.25)
IQR = Q3 - Q1
lower_limit = Q1-(1.5*IQR)
upper_limit = Q3+(1.5*IQR)


# In[25]:


weight_outliers = fish_weight[(fish_weight <lower_limit) | (fish_weight >upper_limit)]
weight_outliers


# In[26]:


plt.figure(figsize = (15,6))
sns.boxplot(fish_data['Length1'])
plt.xticks(rotation = 90)
plt.show()


# In[27]:


fish_length1 = fish_data['Length1']
Q3 = fish_length1.quantile(0.75)
Q1 = fish_length1.quantile(0.25)
IQR = Q3 - Q1
lower_limit = Q1-(1.5*IQR)
upper_limit = Q3+(1.5*IQR)


# In[28]:


length1_outliers = fish_length1[(fish_length1 <lower_limit) | (fish_length1 >upper_limit)]
length1_outliers


# In[29]:


plt.figure(figsize = (15,6))
sns.boxplot(fish_data['Length2'])
plt.xticks(rotation = 90)
plt.show()


# In[30]:


fish_length2 = fish_data['Length2']
Q3 = fish_length2.quantile(0.75)
Q1 = fish_length2.quantile(0.25)
IQR = Q3 - Q1
lower_limit = Q1-(1.5*IQR)
upper_limit = Q3+(1.5*IQR)


# In[31]:


length2_outliers = fish_length2[(fish_length2 <lower_limit) | (fish_length2 >upper_limit)]
length2_outliers


# In[32]:


plt.figure(figsize = (15,6))
sns.boxplot(fish_data['Length3'])
plt.xticks(rotation = 90)
plt.show()


# In[33]:


fish_length3 = fish_data['Length3']
Q3 = fish_length3.quantile(0.75)
Q1 = fish_length3.quantile(0.25)
IQR = Q3 - Q1
lower_limit = Q1-(1.5*IQR)
upper_limit = Q3+(1.5*IQR)


# In[34]:


length3_outliers = fish_length3[(fish_length3 <lower_limit) | (fish_length3 >upper_limit)]
length3_outliers


# In[35]:


plt.figure(figsize = (15,6))
sns.boxplot(fish_data['Height'])
plt.xticks(rotation = 90)
plt.show()


# In[36]:


fish_height = fish_data['Height']
Q3 = fish_height.quantile(0.75)
Q1 = fish_height.quantile(0.25)
IQR = Q3 - Q1
lower_limit = Q1-(1.5*IQR)
upper_limit = Q3+(1.5*IQR)


# In[37]:


height_outliers = fish_height[(fish_height <lower_limit) | (fish_height >upper_limit)]
height_outliers


# In[38]:


plt.figure(figsize = (15,6))
sns.boxplot(fish_data['Width'])
plt.xticks(rotation = 90)
plt.show()


# In[39]:


fish_width = fish_data['Width']
Q3 = fish_width.quantile(0.75)
Q1 = fish_width.quantile(0.25)
IQR = Q3 - Q1
lower_limit = Q1-(1.5*IQR)
upper_limit = Q3+(1.5*IQR)


# In[40]:


width_outliers = fish_width[(fish_width <lower_limit) | (fish_width >upper_limit)]
width_outliers


# In[41]:


fish_data[142:145]


# In[42]:


fish_data_new = fish_data.drop([142,143,145])


# In[43]:


fish_data_new.head()


# In[44]:


fish_data_new.shape


# In[45]:


from sklearn.preprocessing import StandardScaler


# In[46]:


scaler = StandardScaler()


# In[47]:


scaling_columns = ['Weight','Length1','Length2','Length3','Height','Width']
fish_data_new[scaling_columns] = scaler.fit_transform(fish_data_new[scaling_columns])
fish_data_new.describe()


# In[48]:


fish_data_new


# In[49]:


from sklearn.preprocessing import LabelEncoder


# In[50]:


label_encoder = LabelEncoder()


# In[51]:


fish_data_new['Species'] = label_encoder.fit_transform(fish_data_new['Species'].values)


# In[52]:


data_cleaned = fish_data_new.drop("Weight",axis=1)
y = fish_data_new["Weight"]


# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


x_train,x_test,y_train,y_test = train_test_split(data_cleaned,y,test_size = 0.2,random_state = 42)


#  #                                   RandomForestRegressor

# In[55]:


from sklearn.ensemble import RandomForestRegressor


# In[56]:


model = RandomForestRegressor()
model.fit(x_train,y_train)


# In[57]:


y_pred = model.predict(x_test)


# In[58]:


print("Training Accuracy :",model.score(x_train,y_train))
print("Testing Accuracy :",model.score(x_test,y_test))


# #                                       Decision Tree Regressor

# In[59]:


from sklearn.tree import DecisionTreeRegressor
reg1 = DecisionTreeRegressor()


# In[60]:


reg1.fit(x_train,y_train)


# In[61]:


y_pred = reg1.predict(x_test)


# In[62]:


print("Training Accuracy :",reg1.score(x_train,y_train))
print("Testing Accuracy :",reg1.score(x_test,y_test))


# #                                    Linear Regression

# In[63]:


from sklearn.linear_model import LinearRegression
reg2 = LinearRegression()
reg2.fit(x_train,y_train)


# In[64]:


y_pred = reg1.predict(x_test)


# In[65]:


print("Training Accuracy :",reg2.score(x_train,y_train))
print("Testing Accuracy :",reg2.score(x_test,y_test))


# In[66]:


#pip install xgboost


# In[67]:


import xgboost as xgb
# xgb =XGBRegressor()
xgb1 = xgb.XGBRegressor()


# In[68]:


xgb1.fit(x_train,y_train)
xgb_pred = xgb1.predict(x_test)


# In[69]:


print("Training Accuracy : ",xgb1.score(x_train,y_train))
print("Testing Accuracy : ",xgb1.score(x_test,y_test))


# #                                     Deploying the code

# In[70]:


xgb1.save_model("model.json")


# In[71]:


#pip install streamlit


# In[72]:


import streamlit as st


# In[73]:


st.header("Fish Weight Prediction App")
st.text_input("Enter Your Name: ",key="name")


# In[74]:


np.save('classes.npy', label_encoder.classes_)


# In[75]:


label_encoder.classes_ = np.load('classes.npy',allow_pickle=True)


# In[76]:


xgb_best = xgb.XGBRegressor()


# In[77]:


xgb_best.load_model('model.json')


# In[78]:


if st.checkbox('Show Training Dataframe'):
    fish_data


# In[79]:


st.subheader("Please select relevant features of your fish")
left_column,right_column = st.columns(2)
with left_column:
    inp_species = st.radio('Name of fish:',np.unique(fish_data['Species']))


# In[80]:


input_Length1 = st.slider('Vertical Length(cm)',0.0,max(fish_data["Length1"]),1.0)
input_Length2 = st.slider('Diagonal Length(cm)',0.0,max(fish_data["Length2"]),1.0)
input_Length3 = st.slider('Cross Length(cm)',0.0,max(fish_data["Length3"]),1.0)
input_Height = st.slider('Height Length(cm)',0.0,max(fish_data["Height"]),1.0)
input_Width1 = st.slider('Diagonal width Length(cm)',0.0,max(fish_data["Width"]),1.0)


# In[81]:


if st.button('Make Predictions'):
    input_species = label_encoder.transform(np.expand_dims(inp_species,-1))
    inputs = np.expand_dims([int(input_species),input_Length1, input_Length2, input_Length3, input_Height, input_Width],0)
    prediction = xgb_best.predict(inputs)
    print("final pred", np.squeeze(predictions,-1))
    st.write(f"Your fish weight is: {np.squeeze(predictions,-1):.2f}g")


# In[ ]:





# In[ ]:




