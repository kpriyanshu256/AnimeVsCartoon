import os
import cv2
import numpy as np
import pickle

def prepare_data(folder):
    data=[]
    for j in os.listdir(folder):
        for i in os.listdir(folder+'\\'+j):
            img=cv2.imread(folder+'\\'+j+'\\'+i,cv2.IMREAD_COLOR)
            img=cv2.resize(img,(120,80))
            img1 = img[:,:]
            img1=np.array(img1,dtype=np.float64)
            data.append(img1)
    return data


dir=os.getcwd()

A=dir+'\Anime'
C=dir+'\Cartoon'
a=prepare_data(A)
c=prepare_data(C)

data={}
data['anime']=a
data['cartoon']=c

save=open('NewData.pkl','wb')
pickle.dump(data,save)
save.close()



