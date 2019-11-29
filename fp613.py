#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pickle
from IPython.display import Image, display
from random import randint
import imageio


# In[2]:


os.chdir(r'C:\Users\susanthanakrishnan\Documents\ISEN 613\Project\batch_bin')
os.getcwd()
filenames = ["batch_bin\\" + x for x in os.listdir('batch_bin') if '.bin' in x]
filenames[0]


# In[3]:


# Read the image files
data = bytes()
for filename in filenames:
    with open(filename, mode='rb') as file:
        data += file.read()

# Read the class names
with open('batch_bin\\' + 'batches.meta.txt', mode='r') as file:
    txt_content = file.read()

class_names = [x for x in txt_content.splitlines() if len(x.strip()) > 0]
print(class_names) 


# In[4]:


byt_length = len(data)
assert byt_length == 50000*3073, 'Input files not loaded properly, check folder/file.'


# In[5]:


len(data)


# In[6]:


rgb = np.frombuffer(data, dtype=np.uint8)


# In[191]:


rgb2 = rgb.reshape(50000, 3073)
labels = rgb2[:, 0]
features = rgb2[:, 1:]

# To visualise a random image from the dataset
rgb_3d = features.reshape(50000, 3, 1024)
rand_img_num = randint(0, 49999)
lab = labels[rand_img_num]
img1 = rgb_3d[rand_img_num, :, :]
imgt = img1.transpose().reshape(32, 32, 3)
img_grey = np.mean(imgt, axis=2)
img_grey_u8 = np.mean(imgt, axis=2, dtype=np.uint8)
img_grey_floor = np.floor(np.mean(imgt, axis=2))

imageio.imwrite('test1.png', imgt)
imageio.imwrite('grey_test1.png', img_grey)
imageio.imwrite('grey_test1_u8.png', img_grey)
imageio.imwrite('floor_grey_test1.png', img_grey_floor)

display(Image('test1.png'))
display(Image('grey_test1.png'))
display(Image('grey_test1_u8.png'))
display(Image('floor_grey_test1.png'))

print(class_names[lab])

display(img_grey)
display(img_grey_u8)
display(img_grey_floor)


# In[128]:


print(features.shape)
print(labels.shape)


# In[95]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
#X, y = load_iris(return_X_y=True)
X = features[:10000]
y = labels[:10000]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

scaler_test = StandardScaler().fit(X_test)
X_test_scaled = scaler_test.transform(X_test)

import pandas as pd

y_sr = pd.Series(y)
print(np.unique(y, return_counts=True))
print(y_sr.value_counts())

display(X_test.shape)
display(y_test.shape)


# In[72]:


import time
t1 = time.time()
clf = LogisticRegressionCV(cv=5, random_state=0,
                            multi_class='auto', solver='saga', n_jobs=7).fit(X_train_scaled, y_train)
t2 = time.time()
print(t2-t1)


# In[85]:


print(clf.predict(X_scaled[:20, :]))
print(y[:20])
print(X[:2, :])
#print(clf.predict_proba(X[:2, :]))

print(clf.score(X_train_scaled, y_train))
print(clf.score(X_test_scaled, y_test))
#pd.Series(y_test).value_counts()


# In[102]:


import time
t1 = time.time()
clf2 = LogisticRegression(multi_class='auto', solver='liblinear').fit(X_train_scaled[:2000], y_train[:2000])
t2 = time.time()
print(t2-t1)


# In[107]:


print(clf2.score(X_train_scaled[:2000], y_train[:2000]))
print(clf2.score(X_test_scaled[:2000], y_test[:2000]))


# In[87]:


X_train_scaled


# In[111]:


rgb_3d[0]


# In[192]:


feat_mean = np.floor(np.mean(rgb_3d, axis=1))


# In[193]:


feat_mean.shape


# In[194]:


rgb_3d.shape


# In[236]:


imageio.imwrite('grey.png', feat_mean[randint(0,49999)].reshape(32, 32, 1))
display(Image('grey.png'))


# In[315]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
#X, y = load_iris(return_X_y=True)
X = feat_mean
#X = features
y = labels

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

import pandas as pd

y_sr = pd.Series(y)
print(np.unique(y, return_counts=True))
print(y_sr.value_counts())

display(X_test.shape)
display(y_test.shape)


# In[316]:


import time
t1 = time.time()
nx = 10000
x1 = X_train_scaled
y1 = y_train

#x1 = X_train[:nx]
#x1 = features[:nx]
#y1

clf_m = LogisticRegressionCV(cv=5, multi_class='auto', solver='saga', n_jobs=-1).fit(x1, y1)
#clf2 = LogisticRegression(multi_class='auto', solver='saga', n_jobs=7).fit(x1, y1)
#clf3 = LogisticRegressionCV(cv=3, multi_class='auto', solver='saga', n_jobs=7).fit(x1, y1)
#clf4 = LogisticRegressionCV(cv=3, multi_class='auto', solver='saga', n_jobs=7).fit(x1, y1)
t2 = time.time()
print(t2-t1)
# using original & CV=3
print(clf_m.score(x1, y1))
print(clf_m.score(X_test_scaled, y_test))


# In[317]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
#X, y = load_iris(return_X_y=True)
#X = feat_mean[:10000]
X = features
y = labels

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

import pandas as pd

y_sr = pd.Series(y)
print(np.unique(y, return_counts=True))
print(y_sr.value_counts())

display(X_test.shape)
display(y_test.shape)

import time
t1 = time.time()
nx = 10000
x1 = X_train_scaled
y1 = y_train

#x1 = X_train[:nx]
#x1 = features[:nx]
#y1

clf_o = LogisticRegressionCV(cv=5, multi_class='auto', solver='saga', n_jobs=-1).fit(x1, y1)
#clf2 = LogisticRegression(multi_class='auto', solver='saga', n_jobs=7).fit(x1, y1)
#clf3 = LogisticRegressionCV(cv=3, multi_class='auto', solver='saga', n_jobs=7).fit(x1, y1)
#clf4 = LogisticRegressionCV(cv=3, multi_class='auto', solver='saga', n_jobs=7).fit(x1, y1)
t2 = time.time()
print(t2-t1)
# using original & CV=3
print(clf_o.score(x1, y1))
print(clf_o.score(X_test_scaled, y_test))


# In[291]:


# using mean
print(clf2.score(x1, y1))
print(clf2.score(X_test_scaled, y_test))
print(clf2.score(X_test, y_test))


# In[310]:


# using original
print(clf2.score(x1, y1))
print(clf2.score(X_test_scaled, y_test))
print(clf2.score(X_test, y_test))


# In[304]:


# using mean & CV=3
print(clf3.score(x1, y1))
print(clf3.score(X_test_scaled, y_test))
print(clf3.score(X_test, y_test))


# In[312]:


# using original & CV=3
print(clf4.score(x1, y1))
print(clf4.score(X_test_scaled, y_test))
print(clf4.score(X_test, y_test))


# In[262]:


X_test_scaled = scaler_test.transform(X_test)


# In[296]:


features.shape


# In[ ]:




