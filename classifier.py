#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np 
import pandas as pd 
import librosa.display
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[13]:


music_data = pd.read_csv('file.csv') 
music_data.head(5)


# In[14]:


music_data['label'].value_counts()


# In[15]:


path = 'genres_original/blues/blues.00000.wav'
plt.figure(figsize=(14, 5)) 
x, sr = librosa.load(path) 
librosa.display.waveplot(x, sr=sr) 
id.Audio(path) 
  
print("Blue")


# In[16]:


path = 'genres_original/metal/metal.00000.wav'
plt.figure(figsize=(14, 5)) 
x, sr = librosa.load(path) 
librosa.display.waveplot(x, sr=sr,color='orange') 
id.Audio(path) 
  
print("Metal")


# In[17]:


path = 'genres_original/pop/pop.00000.wav'
plt.figure(figsize=(14, 5)) 
x, sr = librosa.load(path) 
librosa.display.waveplot(x, sr=sr,color='purple') 
id.Audio(path) 
  
print("Pop")


# In[18]:


path = 'genres_original/hiphop/hiphop.00000.wav'
plt.figure(figsize=(14, 5)) 
x, sr = librosa.load(path) 
librosa.display.waveplot(x, sr=sr,color='grey') 
id.Audio(path) 
  
print("HipHop")


# In[19]:


import numpy as np 
import seaborn as sns 
  
# Computing the Correlation Matrix 
spike_cols = [col for col in data.columns if 'mean' in col] 
  
# Set up the matplotlib figure 
f, ax = plt.subplots(figsize=(16, 11)); 
  
# Draw the heatmap with the mask and correct aspect ratio 
sns.heatmap(data[spike_cols].corr(), cmap='YlGn') 
  
plt.title('Heatmap for MEAN variables', fontsize = 20) 
plt.xticks(fontsize = 10) 
plt.yticks(fontsize = 10);


# In[20]:


#Data Preprocessing

from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
music_data['label'] = label_encoder.fit_transform(music_data['label'])


# In[21]:


X = music_data.drop(['label','filename'],axis=1) 
y = music_data['label']


# In[22]:


cols = X.columns 
minmax = preprocessing.MinMaxScaler() 
np_scaled = minmax.fit_transform(X) 
  
# new data frame with the new scaled data.  
X = pd.DataFrame(np_scaled, columns = cols)


# In[23]:


from sklearn.model_selection import train_test_split 
  
X_train, X_test, y_train, y_test = train_test_split(X, y,  
                                                    test_size=0.3,  
                                                    random_state=111) 
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[24]:


from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
import catboost as cb 
from xgboost import XGBClassifier 
  
rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0) 
cbc = cb.CatBoostClassifier(verbose=0, eval_metric='Accuracy', loss_function='MultiClass') 
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05) 
  
for clf in (rf, cbc, xgb): 
    clf.fit(X_train, y_train) 
    preds = clf.predict(X_test) 
    print(clf.__class__.__name__,accuracy_score(y_test, preds))


# In[25]:


import tensorflow.keras as keras 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import *
  
model = Sequential() 
  
model.add(Flatten(input_shape=(58,))) 
model.add(Dense(256, activation='relu')) 
model.add(BatchNormalization()) 
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.3)) 
model.add(Dense(10, activation='softmax')) 
model.summary()


# In[26]:


# compile the model 
adam = keras.optimizers.Adam(lr=1e-4) 
model.compile(optimizer=adam, 
             loss="sparse_categorical_crossentropy", 
             metrics=["accuracy"]) 
  
hist = model.fit(X_train, y_train, 
                 validation_data = (X_test,y_test), 
                 epochs = 100, 
                 batch_size = 32)


# In[27]:


test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1) 
print(f"Test accuracy: {test_accuracy}")


# In[28]:


fig, axs = plt.subplots(2,figsize=(10,10)) 
  
# accuracy  
axs[0].plot(hist.history["accuracy"], label="train") 
axs[0].plot(hist.history["val_accuracy"], label="test")     
axs[0].set_ylabel("Accuracy") 
axs[0].legend() 
axs[0].set_title("Accuracy") 
      
# Error  
axs[1].plot(hist.history["loss"], label="train") 
axs[1].plot(hist.history["val_loss"], label="test")     
axs[1].set_ylabel("Error") 
axs[1].legend() 
axs[1].set_title("Error") 
      
plt.show()


# In[ ]:




