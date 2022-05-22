import os
import argparse
import numpy as np
from skimage import io
from skimage.util import img_as_float
from colorspace import Color_Classifier
from texture import Texture_Classfier
from component import ColorComponent_Classfier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

def test(opt, data_path='./BoWFireDataset/dataset/img/', fire_threshold=0.01):
    CC = Color_Classifier(n_neighbors=7)
    CC.train()
    TC = Texture_Classfier(n_neighbors=7, method='default', n_segments=120, max_bins=255, m=40)
    TC.train()
    SC = ColorComponent_Classfier(n_neighbors=9 ,method='default' , n_segments=100, max_bins=255, m=40)
    SC.train()
    test_images = os.listdir(data_path)
    test_images.sort()
    # test_images.reverse()
    if '.DS_Store' in test_images:
        test_images.remove('.DS_Store')
    # random.shuffle(test_images) 
    if not os.path.exists('./results/combined/'):
        os.mkdir('./results/combined/')

    truth = []
    predicted = []
    idx = 0
    for img_name in test_images:
        idx += 1
        img_path = os.path.join(data_path, img_name)
        img = img_as_float(io.imread(img_path))
        img0 = img
        h,w,_ = img.shape
        mask = np.ones(img.shape)

        if opt.color_space:
            color_mask = CC.get_mask(img)
            mask = mask*color_mask
            img_color = img * color_mask
        else:
            img_color = img

        if opt.color_component:
            space_mask = SC.get_mask(img_color)
            img_space = img_color * space_mask
            mask = mask*space_mask
        else:
            img_space = img_color

        if opt.texture:
            texture_mask = TC.get_mask(img_space)
            mask = mask*texture_mask #* color_mask
            img_texture = img_space * texture_mask
        else:
            img_texture = img_space

        img_out = img_texture

        img_out_name = img_name[:-4] + '_out' + img_name[-4:] 
        img_save_path = os.path.join('./results/combined/', img_out_name)
        # io.imsave(img_save_path, img_out.astype(np.uint8))

        is_fire = mask.sum()/h/w > fire_threshold

        plt.figure(figsize=(10,8))
        plt.subplot(2,2,1)
        plt.imshow(img0)
        plt.title('Original Image')
        plt.subplot(2,2,2)
        plt.imshow(img_color)
        plt.title('Color Mask')
        plt.subplot(2,2,3)
        plt.imshow(img_texture)
        plt.title('Texture Mask')
        plt.subplot(2,2,4)
        plt.imshow(img_out)
        plt.title('Fire Mask' + (' (fire)' if is_fire else ' (not fire)'))
        plt.savefig(img_save_path)
        plt.close()

        truth.append(1 if is_fire else 0)
        predicted.append(1 if img_name[0] == 'f' else 0)
        print('[{}/{}]  '.format(idx, len(test_images)) + img_name + ('  fire' if is_fire else '  not fire'))
    
    truth = np.array(truth)
    predicted = np.array(predicted)
    acc = (truth==predicted).sum()/len(test_images)
    print('accuracy: {:.2f}'.format(acc))

    sns.set()
    f,ax = plt.subplots()
    cm = confusion_matrix(truth, predicted)
    cm = pd.DataFrame(cm,columns=["not fire", "fire"],index=["not fire", "fire"])
    sns.heatmap(cm,cmap="YlGnBu_r",fmt="d",annot=True)
    plt.savefig('./results/cm_{:.3f}.jpeg'.format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--color_space', type=bool, default=True, help='whether to use color space classifier')
    parser.add_argument('--color_component', type=bool, default=True, help='whether to use color component classifier')
    parser.add_argument('--texture', type=bool, default=True, help='whether to use texture classifier')
    opt = parser.parse_args()

    test(data_path='./BoWFireDataset/dataset/img/', opt=opt)