
# coding: utf-8

# In[2]:


import sys
print (sys.executable)


# In[3]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
from pandas.plotting import scatter_matrix
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import h5py
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam
from keras import regularizers
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
from sklearn.metrics import (confusion_matrix, auc, roc_curve, cohen_kappa_score, accuracy_score)


# In[4]:


kddCupTrain = pd.read_csv('kddCupTrain.csv',header=None)
print("Shape of kddCupTrain: ",kddCupTrain.shape)
print("There are any missing values: ", kddCupTrain.isnull().values.any())
print(kddCupTrain.head(3))


# In[5]:


kddCupTrain.rename(columns={41:'Class'}, inplace=True)
kddCupTrain['Class'] = np.where(kddCupTrain['Class'] == 'normal.', 0, 1)


# In[6]:


count_classes = pd.value_counts(kddCupTrain['Class'], sort = True)
print(count_classes)


# In[7]:


print(kddCupTrain.describe(percentiles=[]))


# In[8]:


#Droping columns with min=max and std=0
kddCupTrain.drop([7,19], axis=1, inplace=True)


# In[9]:


kddCupTrain = pd.get_dummies(kddCupTrain, columns = [1,2,3])
kddCupTrain.head()


# In[10]:


kddCupTrain.columns


# In[11]:


featuresList = [col for col in kddCupTrain if col != 'Class']
print('featuresList: ',featuresList)
featuresList.remove('2_tftp_u')
featuresList.remove('3_SH')


# In[12]:


kddCupTrain = kddCupTrain[featuresList + ['Class']]
#print('\nkddCup sample: \n')
#kddCupTrain


# In[13]:


kddCupTrain.describe(percentiles=[])


# Exploring the data

# In[14]:


frauds = kddCupTrain[kddCupTrain.Class == 1]
normal = kddCupTrain[kddCupTrain.Class == 0]


# In[15]:


#Standardizing the feature list
#featuresListScale =[0, 4, 5, 8, 9, 10, 12, 14, 15, 16, 17, 18, 22, 23, 31, 32]
scaler = preprocessing.StandardScaler()
scaler.fit(kddCupTrain[featuresList]);


# In[16]:


#featuresListSacle:  [0, 4, 5, 8, 9, 10, 12, 14, 15, 16, 17, 18, 22, 23, 31, 32  


# In[17]:


kddCupTrain[featuresList] = scaler.transform(kddCupTrain[featuresList])


# In[18]:


#Check results f standardization
print('Mean values:')
print(kddCupTrain[featuresList].mean())
print('\nStd values:')
print(kddCupTrain[featuresList].std(ddof=0))


# In[19]:


X_train_split, X_test_split = train_test_split(kddCupTrain, test_size=0.2,stratify=kddCupTrain['Class'],random_state=RANDOM_SEED)


# In[20]:


y_train = X_train_split['Class']
X_train = X_train_split.drop(['Class'], axis=1)

y_test = X_test_split['Class']
X_test = X_test_split.drop(['Class'], axis=1)
print('Train: shape X',X_train.shape,', shape Y',y_train.shape)
print('Test: shape X',X_test.shape,', shape Y',y_test.shape)


# In[21]:


#Selecting rows for normal class to train autoencoders
X_trainNorm = X_train[y_train == 0]
X_trainNorm_val = X_trainNorm.values # Only values, axes labels removed. This is input for the Autoencoder
X_testNorm_val = X_test[y_test == 0].values # The validation data


# In[22]:


print(y_train.shape),print(X_train.shape),print(X_trainNorm_val.shape)


# Creating the model

# In[23]:


#Setting the parameters
input_dim = X_trainNorm_val.shape[1]
layer1_dim = 41
encoder_dim = 20


# Create Tensors

# In[24]:


input_layer = Input(shape=(input_dim, ))

encoder1 = Dense(layer1_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder2 = Dense(encoder_dim, activation="relu")(encoder1)
decoder1 = Dense(layer1_dim, activation='tanh')(encoder2)
decoder2 = Dense(input_dim, activation='linear')(decoder1)
print('input_layer: ',input_layer)
print('encoder1',encoder1)
print('encoder2',encoder2)
print('decoder1',decoder1)
print('decoder2',decoder2)


# In[25]:


#Create autoencoder from the tensors:
autoencoder = Model(inputs=input_layer, outputs=decoder2)
autoencoder.summary()


# In[26]:


#Training the model
nb_epoch = 20
batch_size = 32

#adam = Adam(lr=0.001)
autoencoder.compile(optimizer='Adam', loss='mean_squared_error')

checkpointer = ModelCheckpoint(filepath="model.h5", verbose=0,save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0) # 'patience' number of not improving epochs
history = autoencoder.fit(X_trainNorm_val, X_trainNorm_val,epochs=nb_epoch,batch_size=batch_size,shuffle=True,
                    validation_data=(X_testNorm_val, X_testNorm_val),verbose=1,
                    callbacks=[checkpointer, earlystopping]).history


# In[27]:


#Load the model saved by checkpointer
autoencoder = load_model('model.h5')


# In[28]:


plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');


# Model Evaluation

# In[29]:


#Calculating predictions by the autoencoder
testPredictions = autoencoder.predict(X_test)
X_test.shape,testPredictions.shape


# In[31]:


#Calculate mean squared error
testMSE = mean_squared_error(X_test.transpose(), testPredictions.transpose(),
                              multioutput='raw_values')
error_df = pd.DataFrame({'reconstruction_error': testMSE,'true_class': y_test})
error_df.head()


# In[32]:


error_df.shape


# Reconstruction errors for normal transactions

# In[33]:


fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]
ax.hist(normal_error_df.reconstruction_error.values, bins=10);


# Reconstruction errors for fraudulent transactions

# In[34]:


fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = error_df[error_df['true_class'] == 1]
ax.hist(fraud_error_df.reconstruction_error.values, bins=10);


# Calculate ROC curve and AUC:

# In[35]:


fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# Prediction

# In[36]:


threshold = normal_error_df.reconstruction_error.quantile(q=0.991)
threshold


# In[37]:


#Plot all errors, normal and fraud cases marked, and the threshold:
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()
for name, group in groups:
    if name == 1:
        MarkerSize = 7
        Color = 'orangered'
        Label = 'Fraud'
        Marker = 'd'
    else:
        MarkerSize = 3.5
        Color = 'b'
        Label = 'Normal'
        Marker = 'o'
    ax.plot(group.index, group.reconstruction_error, 
            linestyle='',
            color=Color,
            label=Label,
            ms=MarkerSize,
            marker=Marker)
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend(loc='upper left', bbox_to_anchor=(0.95, 1))
plt.title("Probabilities of fraud for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();


# In[38]:


y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[39]:


cohen_kappa_score(error_df.true_class, y_pred),accuracy_score(error_df.true_class, y_pred)


# Preparing Test data

# In[40]:


kddCupTest = pd.read_csv('kddCupTest.csv', header=None)
print(kddCupTest.head(3))


# In[41]:


kddCupTest.drop([7,19], axis=1, inplace=True)


# In[42]:


kddCupTest = pd.get_dummies(kddCupTest, columns = [1,2,3])
kddCupTest.head()


# In[43]:


kddCupTest = kddCupTest[featuresList]
print('\nkddCup sample: \n')
kddCupTest


# In[44]:


kddCupTest[featuresList] = scaler.transform(kddCupTest[featuresList])
kddCupTest


# In[45]:


len(featuresList)


# In[46]:


#Calculating predictions by the autoencoder
kddtestPredictions = autoencoder.predict(kddCupTest)
kddCupTest.shape,kddtestPredictions.shape


# In[47]:


#Calculate mean squared error
testMSE = mean_squared_error(kddCupTest.transpose(), kddtestPredictions.transpose(),
                              multioutput='raw_values')
result_df = pd.DataFrame({'reconstruction_error': testMSE})
result_df.head()


# In[48]:


result_df.to_csv('filename.csv')


# In[49]:


pwd

