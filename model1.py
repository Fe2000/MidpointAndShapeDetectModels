#!/usr/bin/env python
# coding: utf-8

# # Fernando Estrada¶
# Problem Set 2: Convolutional Neural Networks CS 4143 - Deep Learning
# 
# February 21, 2022

# # Libraries and Imports
# 

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
import sys
import tensorflow as tf


# In[2]:



print("nums of GPus: ", len(tf.config.experimental.list_physical_devices('GPU')))


# # Data and Preprocessing

# In[ ]:


folderP = sys.argv[0]


# In[3]:


path = sys.argv[1]


# In[4]:


data = pd.read_csv(path , nrows= 15000)


# In[5]:


data


# In[6]:


images = [ np.asarray(ImageOps.grayscale(Image.open(x)), dtype = 'float32')/255  for x in data['Files'] ]


# In[7]:


Y = data.iloc[ : , 1:2 ]

enc = OneHotEncoder()

Y = enc.fit_transform(Y).toarray()

Y = pd.DataFrame(Y)


# In[8]:


Y


# In[9]:


images = np.asarray(images)
images = images.reshape(  len(images), 400, 600, 1 )


# In[10]:


x_train , x_test , y_train , y_test = train_test_split(images,Y, test_size=0.2)


# # Model

# In[11]:


model = Sequential()

model.add( Input(shape=(400,600 ,1), dtype="float32"   )  )
model.add(Conv2D(4, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(4, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(4, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(4, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(4, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(4, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add( Dense(3, activation='sigmoid')  )


model.summary()


# In[115]:


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy' , 'Recall' , 'Precision'])


# # Training

# In[116]:


model.fit( x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)


# # Evaluation

# In[14]:


model = keras.models.load_model('ShapeDetectMoreDataGpuV2')


# In[15]:


model.evaluate(x_test,y_test)

