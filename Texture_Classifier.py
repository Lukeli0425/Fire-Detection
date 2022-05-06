import numpy as np
from distutils.ccompiler import new_compiler
from skimage import io
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor

# Texture_Classfier模型
class Texture_Classfier:
    def __init__(self,image_path,fire_feature,normal_feature,smoke_feature,n_segments):
        # params
        self.n_points = 8
        self.radius = 1
        self.n_segments = n_segments

        self.image = img_as_float(io.imread(image_path))
        self.fire_feature = fire_feature
        self.normal_feature = normal_feature
        self.smoke_feature = smoke_feature
        

    def LBP(self,image):
        image_gray = rgb2gray(image)
        lbp = local_binary_pattern(image_gray,self.n_points,self.radius)
        self.LBP_image = lbp

    def Superpixel(self, sigma=5):
        # segments map
        segments = slic(self.image, n_segments=self.n_segments, sigma=5)
        # num of superpixels
        SP_num = len(np.unique(segments))
        return segments,SP_num

    def generate_mask(self):
        self.LBP(self.image)
        mask = np.zeros(self.image.shape)
        segments,SP_num = self.Superpixel()
        self.segments = segments
        cnt = 0
        model = joblib.load('Texture_KNN.model')

        for idx in range(SP_num):
            # shape (1,n) n is the size of superpixel
            superpixel = self.LBP_image[segments==idx]
            max_bins = 256
            feature,_ = np.histogram(superpixel,bins=max_bins, range=(0, 255),density=True)
            predict = model.predict([feature])
            print("predict = ",predict)
            print("predict输入:",feature.shape)
            # L1,L2,L3 = np.linalg.norm(feature-self.fire_feature),np.linalg.norm(feature-self.smoke_feature),np.linalg.norm(feature-self.normal_feature)
            # print(L1,L2,L3)
            if predict==0:
                cnt += 1
                mask[segments==idx] = 1
        print("debug here::",cnt,SP_num)
        return mask

if __name__ == "__main__":
    image_path = '/Volumes/Samsung_T5/数字图像处理大作业/BoWFireDataset/dataset/img/not_fire046.png'
    fire_feature = np.load('fire_feature.npy')
    smoke_feature = np.load('smoke_feature.npy')
    normal_feature = np.load('normal_feature.npy')
    TC = Texture_Classfier(image_path=image_path,fire_feature=fire_feature,normal_feature=normal_feature,smoke_feature=smoke_feature,n_segments=10)
    Mask = TC.generate_mask()
    plt.figure
    plt.subplot(131)
    plt.title('Fire Mask')
    plt.imshow(Mask)
    plt.subplot(132)
    plt.title('image')
    plt.imshow(TC.image)
    plt.subplot(133)
    plt.title('image and segments')
    plt.imshow(mark_boundaries(TC.image, TC.segments))
    plt.show()
