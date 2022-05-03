import os
import skimage
import numpy as np
import random 
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier

def color_train(data_path='./BoWFireDataset/train/'):
    train_images = os.listdir(data_path)
    if '.DS_Store' in train_images:
        train_images.remove('.DS_Store')
    X = [] # pixels
    Y = [] # labels
    for img_name in train_images:
        img_path = os.path.join(data_path, img_name)
        img = skimage.io.imread(img_path)
        h,w,_ = img.shape
        img_ycbcr = skimage.color.convert_colorspace(img, 'rgb', 'ycbcr')

        for i in range(0,h):
            for j in range(0,w):
                X.append(img_ycbcr[i,j,:])
                if img_name[0] == 'f': # fire
                    Y.append(1)
                else:
                    Y.append(0)
    X = np.array(X)
    Y = np.array(Y)
    model = KNeighborsRegressor(n_neighbors=30)
    # model = LogisticRegression()
    model.fit(X, Y)
    return model

def color_test(model, data_path='./BoWFireDataset/dataset/img/'):
    if not os.path.exists('./color_result/'):
        os.mkdir('./color_result/')
    test_images = os.listdir(data_path)
    if '.DS_Store' in test_images:
        test_images.remove('.DS_Store')
    random.shuffle(test_images)
    for img_name in test_images:
        img_path = os.path.join(data_path, img_name)
        img = skimage.io.imread(img_path)
        h,w,d = img.shape
        img_ycbcr = skimage.color.convert_colorspace(img, 'rgb', 'ycbcr')
        X_test = []
        prds = np.array(model.predict(img_ycbcr.reshape(-1,3)))
        prds = prds.reshape(h,w)
        print(prds.shape)

        img = np.array([img[:,:,0]*prds,img[:,:,1]*prds,img[:,:,2]*prds])
        print(img.shape)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)        
        print(img.shape)
        img_save_path = os.path.join('./color_result/', img_name)
        skimage.io.imsave(img_save_path, img)


if __name__ == "__main__":
    model = color_train()
    color_test(model)