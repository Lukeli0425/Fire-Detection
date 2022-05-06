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
    def __init__(self, n_segments=15, dataset_path='./BoWFireDataset/train/', n_neighbors=11):
        # params
        self.n_points = 8
        self.radius = 1
        self.n_segments = n_segments
        self.dataset_path = dataset_path
        self.n_neighbors = n_neighbors
        self.model = joblib.load('./models/Texture_KNN.model')

    def LBP(self):
        image_gray = rgb2gray(self.image)
        lbp = local_binary_pattern(image_gray, self.n_points, self.radius)
        self.LBP_image = lbp

    def Superpixel(self, sigma=5):
        # segments map
        segments = slic(self.image, n_segments=self.n_segments, compactness=40)  
        # num of superpixels
        SP_num = np.unique(segments).tolist()

        return segments,SP_num

    def train(self, radius=1, n_points=8):
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric='manhattan')
        X = []
        images, train_label = loadPicture(dataset_path=self.dataset_path)
        print("Training Texture KNN...")
        Y = np.array(train_label)
        for idx in range(len(images)):
            image = images[idx]
            X.append(Extract_LBP_Feature(image,radius=radius,n_points=n_points))
        
        self.model.fit(X, Y)
        joblib.dump(self.model,'./models/Texture_KNN.model')

    def get_mask(self, image):
        self.image = image
        self.LBP()
        mask = np.zeros(self.image.shape)
        segments, SP_num = self.Superpixel()
        self.segments = segments
        cnt = 0

        for idx in SP_num:
            # shape (1,n) n is the size of superpixel
            superpixel = self.LBP_image[segments==idx]
            max_bins = 256
            feature,_ = np.histogram(superpixel,bins=max_bins, range=(0, 255),density=True)
            predict = self.model.predict(feature.reshape(1,-1))
            if predict[0]==2:
                cnt += 1
                mask[segments==idx] = 1
        return mask

    def test(self, data_path='./BoWFireDataset/dataset/img/', fire_threshold=0.01):
        total_correct = 0
        if not os.path.exists('./results/texture/'):
            os.mkdir('./results/texture/')
        test_images = os.listdir(data_path)
        if '.DS_Store' in test_images:
            test_images.remove('.DS_Store')

        for img_name in test_images:
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
            plt.imshow((img * mask).astype(np.uint8))
            plt.title('Texture Mask')
            plt.subplot(1,3,3)
            plt.imshow(mark_boundaries(TC.image, TC.segments))
            plt.title('Segments')
            plt.savefig(img_save_path)
            plt.close()

            is_fire = mask.sum()/h/w > fire_threshold
            if img_name[0] == 'f' and is_fire:
                total_correct += 1
            elif img_name[0] == 'n' and (not is_fire):
                total_correct += 1

        acc = total_correct/len(test_images)
        print('accuracy: {:.2f}'.format(acc))
        return acc

if __name__ == "__main__":
    image_path = './BoWFireDataset/dataset/img/fire033.png'
    TC = Texture_Classfier()
    TC.train()
    TC.test()
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
