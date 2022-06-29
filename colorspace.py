import os
import skimage
import numpy as np
from sklearn.naive_bayes import GaussianNB
from skimage.color import convert_colorspace
import joblib

class ColorSpace_Classifier:
    def __init__(self, 
                 dataset_path='./BoWFireDataset/train/', 
                 model_path='./models/Color_Space_Bayes.model', 
                 n_neighbors=11):

        self.dataset_path = dataset_path
        self.model_path = model_path
        self.n_neighbors = n_neighbors
        try:
            self.model = joblib.load(self.model_path)
            print('Successfully loaded ColorSpace model.')
        except:
            print("Warning: Failed loading ColorSpace model!")

    def train(self):
        self.model = GaussianNB()
        train_images = os.listdir(self.dataset_path)
        if '.DS_Store' in train_images:
            train_images.remove('.DS_Store')
        X = [] # pixels
        Y = [] # labels
        print("Training Color Space Naive Bayes...")
        for img_name in train_images:
            img_path = os.path.join(self.dataset_path, img_name)
            img = skimage.io.imread(img_path)
            h,w,_ = img.shape
            img_ycbcr = convert_colorspace(img, 'rgb', 'ycbcr')

            for i in range(0,h):
                for j in range(0,w):
                    X.append(img_ycbcr[i,j,:])
                    if img_name[0] == 'f': # fire
                        Y.append(2)
                    elif img_name[0] == 's':
                        Y.append(1)
                    else:
                        Y.append(0)

        X = np.array(X)
        Y = np.array(Y)
        
        self.model.fit(X, Y)
        joblib.dump(self.model, self.model_path)
        return self.model

    def get_mask(self, img):
        """Generate fire mask of the input image with the saved KNN model trained by colar space mathod."""
        h,w,_ = img.shape
        img_ycbcr = convert_colorspace(img, 'rgb', 'ycbcr')
        prds = np.array(self.model.predict(img_ycbcr.reshape(-1,3)))
        mask = prds.reshape(h,w)
        mask = np.array(mask/2, dtype=int)
        mask = np.array([mask, mask, mask])
        mask = np.swapaxes(mask, 0, 2)
        mask = np.swapaxes(mask, 0, 1)

        return mask

    def test(self, fire_threshold=0.15, data_path='./BoWFireDataset/dataset/img/'):
        total_correct = 0
        if not os.path.exists('./results/colorspace/'):
            os.mkdir('./results/colorspace/')
        test_images = os.listdir(data_path)
        test_images.sort()
        if '.DS_Store' in test_images:
            test_images.remove('.DS_Store')

        idx = 0
        for img_name in test_images:
            idx += 1
            img_path = os.path.join(data_path, img_name)
            img = skimage.io.imread(img_path)
            h,w,_ = img.shape
            mask = self.get_mask(img)
            # img_out = img * mask
            img_out = img
            img_out[mask==0] = 0
            img_out_name = img_name[:-4] + '_out' + img_name[-4:] 
            img_save_path = os.path.join('./results/colorspace/', img_out_name)
            skimage.io.imsave(img_save_path, img_out.astype(np.uint8))

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
    CC = ColorSpace_Classifier()
    CC.train()
    CC.test()