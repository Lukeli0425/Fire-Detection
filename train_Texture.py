# coding=utf-8
# 导入相应的python包
import os
import numpy as np
from distutils.ccompiler import new_compiler
from skimage import io
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

def loadPicture(dataset_path = 'BoWFireDataset/train/'):
    images = []
    train_label = []
    image_paths = os.listdir(dataset_path)
    for image_name in image_paths:
        images.append(img_as_float(io.imread(dataset_path+image_name)))
        if image_name[0] == 'f':
            train_label.append('f')
        elif image_name[0] == 'n':
            train_label.append('n')
        elif image_name[0] == 's':
            train_label.append('s')
    return images,train_label
        
def Extract_LBP_Feature(image,radius=1,n_points=8):
    # LBP特征提取
    # radius = 1  # LBP算法中范围半径的取值
    # n_points = 8 * radius # 领域像素点数
    image_gray = rgb2gray(image)
    lbp = local_binary_pattern(image_gray,n_points,radius)
    max_bins = 256
    feature,_ = np.histogram(lbp,bins=max_bins, range=(0, 255),density=True)
    return feature


def train(dataset_path = 'BoWFireDataset/train/',radius=1,n_points=8):
    fire_num = 0
    smoke_num = 0
    normal_num = 0
    fire_feature = np.zeros((256))
    smoke_feature = np.zeros((256))
    normal_feature = np.zeros((256))
    images,train_label = loadPicture(dataset_path=dataset_path)
    print("Begin training......")
    for idx in range(len(images)):
        print('training:%s/%s'%(idx,len(images)))
        image = images[idx]
        label = train_label[idx]
        if label == 'f':
            fire_feature += Extract_LBP_Feature(image,radius=radius,n_points=n_points)
            fire_num += 1
        if label == 's':
            smoke_feature += Extract_LBP_Feature(image,radius=radius,n_points=n_points)
            smoke_num += 1
        if label == 'n':
            normal_feature += Extract_LBP_Feature(image,radius=radius,n_points=n_points)
            normal_num += 1     
    fire_feature,smoke_feature,normal_feature = fire_feature/fire_num,smoke_feature/smoke_num,normal_feature/normal_num
    np.save('fire_feature.npy',fire_feature)
    np.save('smoke_feature.npy',smoke_feature)
    np.save('normal_feature.npy',normal_feature)
    
if __name__ == "__main__":
    train(dataset_path='BoWFireDataset/train/')