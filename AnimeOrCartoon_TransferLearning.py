from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file):
    file=open(file,'rb')
    data=pickle.load(file)
    file.close()
    return data

def process_data(d):
    A=d['anime']
    C=d['cartoon']
    a=len(A)
    c=len(C)
    m=a+c
    y=np.zeros(shape=(m,2))
    y[0:a,0]=1
    y[a:,1]=1
    X=[]
    for i in A:
        X.append(i)
    for i in C:
        X.append(i)
    X=np.array(X)
    return X,y

file='NewData.pkl'
data=load_data(file)
X,y=process_data(data)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

img_width=80
img_height=120

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

for layer in model.layers[:5]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

model = Model(inputs = model.input, outputs = predictions)
model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=96, epochs=40)
score = model.evaluate(X_test, y_test, batch_size=96)
model.save("VGG_model.h5")
