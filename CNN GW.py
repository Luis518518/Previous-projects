import seaborn as sns
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,SimpleRNN,BatchNormalization,LSTM,Input
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from keras.optimizers import RMSprop 
import numpy as np
import sys
import pylab
import pandas as pd
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

DatL=np.load('DatL.npy')
DatH=np.load('DatH.npy')
DattestL=np.load('DattestL.npy')
DattestH=np.load('DattestH.npy')

aux=[]
auxt=[]
for i in range(0,1098):
            if i % 6 == 0 or i % 6 == 1 or i % 6==2:
                aux.append(0)
            else:
                aux.append(1)
for i in range(0,366):
            if i % 6 == 0 or i % 6 == 1 or i % 6==2:
                auxt.append(0)
            else:
                auxt.append(1)
print('DatL')
print(DatL.shape)
#print(DatL[1097])

print('DatH')
print(DatH.shape)
#print(DatH[1097])

print('DattestL')
print(DattestL.shape)
#print(DattestL[1])

print('DattestH')
print(DattestH.shape)
#print(DattestH[1])

plt.figure()
plt.imshow(DattestL[5])
plt.grid(True)

#DatH=DatH/255.0
#DattestH=DattestH/255.0

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 50,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3),activation='sigmoid'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3),activation='sigmoid'))
model.add(Flatten())
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))  
  
model.summary()  

model.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])


DatL=DatL.reshape(-1,28,50,1)
history = model.fit(DatL, aux, batch_size = 20, epochs = 3)

DattestL = DattestL.reshape(-1,28,50,1)
predictionL=model.predict(x=DattestL, batch_size=20,verbose=0)

predictionL = np.around(predictionL) 
print('PredictionL1')
print(type(predictionL))
print(predictionL.shape)
print(predictionL)
print(len(predictionL))
test_loss, test_acc = model.evaluate(DattestL, auxt)
print('Accuracy L1', test_acc)
cm=confusion_matrix(y_true=auxt,y_pred=predictionL)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm_labels=['Ruido', 'Ruido + GW']    
#plot_confusion_matrix(cm=cm,classes=cm_labels,title='Matriz de confusion L1')


DatH=DatH.reshape(-1,28,50,1)
history = model.fit(DatH, aux, batch_size = 20, epochs = 3)
DattestH = DattestH.reshape(-1,28,50,1)
predictionH=model.predict(x=DattestH, batch_size=20,verbose=0)

test_loss, test_acc = model.evaluate(DattestH, auxt)
print('Accuracy H1', test_acc)

predictionH = np.around(predictionH) 
cmH=confusion_matrix(y_true=auxt,y_pred=predictionH)
plot_confusion_matrix(cm=cmH,classes=cm_labels,title='Matriz de confusion H1')

plt.figure()
plt.imshow(DattestL[15])
plt.grid(True)