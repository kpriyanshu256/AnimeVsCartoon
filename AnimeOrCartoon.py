import pickle
import numpy as np
import tflearn
from sklearn.model_selection import train_test_split
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Function to load data from pickled file
def load_data(file):
    file=open(file,'rb')
    data=pickle.load(file)
    file.close()
    return data

# Function to segregate data
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
    
# Model
def cnn(X_train,y_train,X_test,y_test):
    # Image processing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # Convolutional network building
    network = input_data(shape=[None, 80, 120, 3],data_preprocessing=img_prep,data_augmentation=img_aug)
    network = conv_2d(network, 32, 5, activation='relu')
    network = max_pool_2d(network, 3)
    network = conv_2d(network, 64, 5, activation='relu')
    network = conv_2d(network, 64, 5, activation='relu')
    network = max_pool_2d(network, 3)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)
    
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(X_train, y_train, n_epoch=40, shuffle=True, validation_set=(X_test, y_test),show_metric=True, batch_size=96, run_id='AC')
    
    model.save('CNN.tflearn')
    
    
if __name__=='__main__':
    file='NewData.pkl'
    data=load_data(file)
    X,y=process_data(data)
    X=X[0:6400,:,:,:]
    y=y[0:6400,:]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    cnn(X_train,y_train,X_test,y_test)
