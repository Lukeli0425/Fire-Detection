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
from sklearn.neighbors import KNeighborsClassifier

def loadPicture(dataset_path='./BoWFireDataset/train/'):
    images = []
    train_label = []
    image_paths = os.listdir(dataset_path)
    for image_name in image_paths:
        images.append(img_as_float(io.imread(dataset_path+image_name)))
        if image_name[0] == 'f':
            train_label.append(2)
        elif image_name[0] == 'n':
            train_label.append(0)
        elif image_name[0] == 's':
            train_label.append(1)
    return images, train_label
    
def Extract_LBP_Feature(image, radius=1, n_points=8, max_bins=255, method='default'):
    # LBP特征提取
    # radius = 1  # LBP算法中范围半径的取值
    n_points = 8 * radius # 领域像素点数
    image_gray = rgb2gray(image)
    lbp = local_binary_pattern(image_gray, n_points, radius, method=method)
    feature, _ = np.histogram(lbp,bins=max_bins, range=(0, 255), density=True)
    # print('debug here:',feature.sum())
    # assert 1==0
    return feature

# Texture_Classfier模型
class Texture_Classfier:
    def __init__(self, 
                 dataset_path='./BoWFireDataset/train/',
                 model_path='./models/Texture_KNN.model',
                 n_neighbors=11,
                 method='default',
                 n_segments=50, 
                 m=40,
                 max_bins=32):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.n_points = 8
        self.radius = 1
        self.method = method
        self.n_segments = n_segments
        self.m = m
        self.n_neighbors = n_neighbors
        self.max_bins = max_bins
        try:
            self.model = joblib.load(self.model_path)
            print('Successfully loaded Texture model.')
        except:
            print("Warning: Failed loading Texture model!")


    def LBP(self):
        image_gray = rgb2gray(self.image)
        lbp = local_binary_pattern(image_gray, self.n_points, self.radius, method=self.method)
        self.LBP_image = lbp

    def Superpixel(self, sigma=5):
        # segments map
        segments = slic(self.image, n_segments=self.n_segments, compactness=self.m)
        # num of superpixels
        SP_num = np.unique(segments).tolist()
        return segments, SP_num

    def train(self, radius=1, n_points=8):
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights='distance', metric='manhattan')
        X = []
        images, train_label = loadPicture(dataset_path=self.dataset_path)

        print("Training Texture KNN...")
        Y = np.array(train_label)
        for idx in range(len(images)):
            image = images[idx]
            X.append(Extract_LBP_Feature(image, radius=radius, n_points=n_points, max_bins=self.max_bins, method=self.method))
        
        self.model.fit(X, Y)
        joblib.dump(self.model,'./models/Texture_KNN.model')

    def get_mask(self, image):
        self.image =image
        self.LBP()
        mask = np.zeros(self.image.shape)
        segments, SP_num = self.Superpixel()
        self.segments = segments
        cnt = 0

        for idx in SP_num:
            # shape (1,n) n is the size of superpixel
            superpixel = self.LBP_image[segments==idx]
            feature,_ = np.histogram(superpixel, bins=self.max_bins, range=(0, 255), density=True)
            predict = self.model.predict(feature.reshape(1,-1))
            if predict[0]==2:
                cnt += 1
                mask[segments==idx] = 1
        return mask

    def test(self, data_path, fire_threshold=0.01):
        total_correct = 0
        if not os.path.exists('./results/texture/'):
            os.mkdir('./results/texture/')
        test_images = os.listdir(data_path)
        test_images.sort()
        if '.DS_Store' in test_images:
            test_images.remove('.DS_Store')

        idx = 0
        for img_name in test_images:
            idx += 1
            img_path = os.path.join(data_path, img_name)
            img = io.imread(img_path)
            h,w,_ = img.shape
            mask = self.get_mask(img)
            img_out = img * mask
            img_out_name = img_name[:-4] + '_out' + img_name[-4:] 
            img_save_path = os.path.join('./results/texture/', img_out_name)
            # io.imsave(img_save_path, img_out.astype(np.uint8))

            plt.figure(figsize=(10,5))
            plt.subplot(1,3,1)
            plt.imshow(self.image.astype(np.uint8))
            plt.title('Original Image')
            plt.subplot(1,3,2)
            plt.imshow(img_out.astype(np.uint8))
            plt.title('Texture Mask')
            plt.subplot(1,3,3)
            plt.imshow(mark_boundaries(TC.image, TC.segments).astype(np.uint8))
            plt.title('Segments')
            plt.savefig(img_save_path)
            plt.close()

            is_fire = mask.sum()/h/w > fire_threshold
            if img_name[0] == 'f' and is_fire:
                total_correct += 1
            elif img_name[0] == 'n' and (not is_fire):
                total_correct += 1
            print('[{}/{}]  '.format(idx, len(test_images)) + img_name + ('  fire' if is_fire else '  not fire'))

        acc = total_correct/len(test_images)
        print('accuracy: {:.2f}'.format(acc))
        return acc

if __name__ == "__main__":
    # image_path = './BoWFireDataset/dataset/img/fire033.png'
    image_path = "./results/color/"
    TC = Texture_Classfier()
    TC.train()
    TC.test(data_path=image_path)
    img = img_as_float(io.imread(image_path))
    Mask = TC.get_mask(image=img)

    # plt.subplot(131)
    # plt.title('Fire Mask')
    # plt.imshow(Mask*img)
    # plt.subplot(132)
    # plt.title('image')
    # plt.imshow(TC.image)
    # plt.subplot(133)
    # plt.title('image and segments')
    # plt.imshow(mark_boundaries(TC.image, TC.segments))
    # plt.show()
