import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import joblib
from train_Texture import Extract_LBP_Feature
from sklearn.neighbors import KNeighborsRegressor

def loadPicture(dataset_path='./BoWFireDataset/train/'):
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
    
def Extract_LBP_Feature(image, radius=1, n_points=8):
    # LBP特征提取
    # radius = 1  # LBP算法中范围半径的取值
    # n_points = 8 * radius # 领域像素点数
    image_gray = rgb2gray(image)
    lbp = local_binary_pattern(image_gray,n_points,radius)
    max_bins = 256
    feature,_ = np.histogram(lbp,bins=max_bins, range=(0, 255),density=True)
    return feature

# Texture_Classfier模型
class Texture_Classfier:
    def __init__(self, n_segments=5, dataset_path='./BoWFireDataset/train/'):
        # params
        self.n_points = 8
        self.radius = 1
        self.n_segments = n_segments
        self.dataset_path = dataset_path

    def LBP(self):
        image_gray = rgb2gray(self.image)
        lbp = local_binary_pattern(image_gray,self.n_points,self.radius)
        self.LBP_image = lbp

    def Superpixel(self, sigma=5):
        # segments map
        segments = slic(self.image, n_segments=self.n_segments, sigma=5)  
        # num of superpixels
        SP_num = np.unique(segments).tolist()

        return segments,SP_num

    def train(self, radius=1, n_points=8):
        X = []
        Y = []
        # Y 0:fire 1:smoke 2:normal   
        images,train_label = loadPicture(dataset_path=self.dataset_path)
        print("Training Texture KNN...")
        for idx in range(len(images)):
            image = images[idx]
            label = train_label[idx]
            X.append(Extract_LBP_Feature(image,radius=radius,n_points=n_points))
            if label == 'f':
                Y.append(0)
            elif label == 's':
                Y.append(1)
            elif label == 'n':
                Y.append(2)
        model = KNeighborsRegressor(n_neighbors=15)
        model.fit(X, Y)
        joblib.dump(model,'./models/Texture_KNN.model')

    def get_mask(self, image):
        self.image = image
        self.LBP()
        mask = np.zeros(self.image.shape)
        segments,SP_num = self.Superpixel()
        self.segments = segments
        cnt = 0
        model = joblib.load('./models/Texture_KNN.model')

        for idx in SP_num:
            # shape (1,n) n is the size of superpixel
            superpixel = self.LBP_image[segments==idx]
            max_bins = 256
            feature,_ = np.histogram(superpixel,bins=max_bins, range=(0, 255),density=True)
            predict = model.predict(feature.reshape(1,-1))
            if predict==0:
                print('FFF')
                cnt += 1
                mask[segments==idx] = 1
        return mask

if __name__ == "__main__":
    image_path = './BoWFireDataset/dataset/img/fire000.png'
    TC = Texture_Classfier(n_segments=12)
    TC.train()
    img = img_as_float(io.imread(image_path))
    Mask = TC.get_mask(image=img)
    plt.figure
    plt.subplot(131)
    plt.title('Fire Mask')
    plt.imshow(Mask*img)
    plt.subplot(132)
    plt.title('image')
    plt.imshow(TC.image)
    plt.subplot(133)
    plt.title('image and segments')
    plt.imshow(mark_boundaries(TC.image, TC.segments))
    plt.show()
