#!/usr/bin/env python
# coding: utf-8

# # Fernando Estrada
# Problem Set 2: Convolutional Neural Networks
# CS 4143 - Deep Learning
# 
# February 21, 2022

# # Libraries and Imports

# In[1]:


from keras.models import *
from keras.layers import *
from keras import Input
from keras import Sequential
from tensorflow import keras
from keras import metrics

import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np                      

from PIL import Image, ImageOps
from matplotlib import pyplot as plt    
import cv2
import math
import tensorflow as tf

import sys


# In[2]:


print("nums of GPus: ", len(tf.config.experimental.list_physical_devices('GPU')))


# # Data and Preprocessing

# In[ ]:


folder = sys.argv[0]


# In[3]:


path = sys.argv[1]


# In[4]:


data = pd.read_csv(path , nrows= 15000)


# In[5]:


data


# In[6]:


images = [ np.asarray(ImageOps.grayscale(Image.open(x)), dtype = 'float32')/255  for x in data['Files'] ]


# In[7]:


Y1 = data.iloc[ : , 2:3 ]


Y1 = pd.DataFrame(Y1)


# In[8]:


Y1


# In[9]:


xList = []
lList = []
allList = []
for x in range(15000):
    
    temp = Y1['Centers'][x].split(",")
    temp[0] = temp[0].replace("(" , "")
    temp[1] = temp[1].replace(")" , "")
    temp[1] = temp[1].replace(" " , "")
    
    xList.append(temp[0])
    lList.append(temp[1])
    tempList = [temp[0] , temp[1]]
    allList.append(tempList)
        


# In[10]:


Y1['x1'] = xList
Y1['y1'] = lList
Y1 = Y1.drop(['Centers'], axis=1)


# In[11]:


Y1 = Y1.astype(float)


# In[12]:


Y1


# In[13]:


maxNums = Y1.max(axis = 0)
maxNums


# In[14]:


Y1['x1'] = Y1['x1']/maxNums[0]
Y1['y1'] = Y1['y1']/maxNums[1]


# In[15]:


Y1


# In[16]:


Y1 = Y1.astype(float)


# In[17]:


images = np.asarray(images)
images = images.reshape(  len(images), 400, 600, 1 )


# In[18]:


x_train , x_test , y_train , y_test = train_test_split(images,Y1, test_size=0.2)


# # Model

# In[26]:


model = Sequential()

model.add( Input(shape=(400,600 ,1), dtype="float32"   )  )
model.add(Conv2D(2, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(6, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(12, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(24, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add( Dense(2, activation='linear')  )


model.summary()


# In[27]:


model.compile(optimizer="adam",loss="mean_squared_error")


# # Training

# In[28]:


model.fit( x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)


# # Evaluation

# In[29]:


model.evaluate(x_test,y_test)

