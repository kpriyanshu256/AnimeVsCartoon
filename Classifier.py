from keras.models import load_model
import numpy as np
import cv2
    
img_file='Test5.jpg'
img=cv2.imread(img_file,cv2.IMREAD_COLOR)
img=cv2.resize(img,(120,80))
img1 = img[:,:]
img1=np.array(img1,dtype=np.float64)
img1=np.expand_dims(img1,axis=0)

cnn=load_model("VGG_model.h5")
score = cnn.predict(img1)