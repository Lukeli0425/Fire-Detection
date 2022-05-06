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
import joblib

def color_train(data_path='./BoWFireDataset/train/', n_neighbors=20, model_path='./models/Color_KNN.model'):
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
    
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X, Y)
    joblib.dump(model, model_path)
    return model

def color_test(model_path='./models/Color_KNN.model', data_path='./BoWFireDataset/dataset/img/', fire_threshold=0.1):
    total_correct = 0
    if not os.path.exists('./color_result/'):
        os.mkdir('./color_result/')
    test_images = os.listdir(data_path)
    if '.DS_Store' in test_images:
        test_images.remove('.DS_Store')
    random.shuffle(test_images)

    for img_name in test_images:
        img_path = os.path.join(data_path, img_name)
        img = skimage.io.imread(img_path)
        h,w,_ = img.shape
        mask = color_mask(img, model_path=model_path)

        img_out = np.array([img[:,:,0]*mask,img[:,:,1]*mask,img[:,:,2]*mask])
        img_out = np.swapaxes(img_out, 0, 2)
        img_out = np.swapaxes(img_out, 0, 1)
        img_out_name = img_name[:-4] + '_out' + img_name[-4:] 
        img_save_path = os.path.join('./color_result/', img_out_name)
        skimage.io.imsave(img_save_path, img_out.astype(np.uint8))

        is_fire = mask.sum()/h/w > fire_threshold
        if img_name[0] == 'f' and is_fire:
            total_correct += 1
        elif img_name[0] == 'n' and (not is_fire):
            total_correct += 1

    acc = total_correct/len(test_images)
    print('accuracy: {:.2f}'.format(acc))
    return acc

def color_mask(img, model_path='./models/Color_KNN.model'):
    """Generate fire mask of the input image with the saved KNN model trained by colar space mathod."""
    model = joblib.load(model_path)
    h,w,_ = img.shape
    img_ycbcr = skimage.color.convert_colorspace(img, 'rgb', 'ycbcr')
    prds = np.array(model.predict(img_ycbcr.reshape(-1,3)))
    mask = prds.reshape(h,w)

    return mask


if __name__ == "__main__":
    color_train()
    color_test()