import os
import argparse
import numpy as np
from skimage import io
from skimage.util import img_as_float
from colorspace import ColorSpace_Classifier
from texture import Texture_Classfier
from component import ColorComponent_Classfier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

def test(opt, data_path='./BoWFireDataset/dataset/img/', fire_threshold=0.01):
    if opt.color_space=='True':
        CS = ColorSpace_Classifier(n_neighbors=7)
        # CS.train()
    if opt.color_component=='True':
        CC = ColorComponent_Classfier(n_neighbors=9 ,method='default' , n_segments=100, max_bins=255, m=40)
        # CC.train()
    if opt.texture=='True':
        TC = Texture_Classfier(n_neighbors=7, method='default', n_segments=120, max_bins=255, m=40)
        # TC.train()

    test_images = os.listdir(data_path)
    test_images.sort()
    if '.DS_Store' in test_images:
        test_images.remove('.DS_Store')

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

        if opt.color_space=='True':
            cs_mask = CS.get_mask(img)
            mask = mask*cs_mask
            img_cs = img * cs_mask
        else:
            img_cs = img

        if opt.color_component=='True':
            cc_mask = CC.get_mask(img_cs)
            img_cc = img_cs * cc_mask
            mask = mask*cc_mask
        else:
            img_cc = img_cs

        if opt.texture=='True':
            tc_mask = TC.get_mask(img_cc)
            mask = mask*tc_mask #* cs_mask
            img_tc = img_cc * tc_mask
        else:
            img_tc = img_cc

        img_out = img_tc

        img_out_name = img_name[:-4] + '_out' + img_name[-4:] 
        img_save_path = os.path.join('./results/combined/', img_out_name)
        # io.imsave(img_save_path, img_out.astype(np.uint8))

        is_fire = mask.sum()/h/w > fire_threshold

        plt.figure(figsize=(10,7))
        plt.subplot(2,2,1)
        plt.imshow(img0)
        plt.axis('off')
        plt.title('0. Original Image')
        plt.subplot(2,2,2)
        plt.imshow(img_cs)
        plt.axis('off')
        plt.title('1. After ColorSpace Mask')
        plt.subplot(2,2,3)
        plt.imshow(img_tc)
        plt.axis('off')
        plt.title('2. After ColorComponent Mask')
        plt.subplot(2,2,4)
        plt.imshow(img_out)
        plt.axis('off')
        plt.title('3. After Texture Mask ' + ('(fire)' if is_fire else '(not fire)'))
        plt.savefig(img_save_path, bbox_inches='tight')
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
    plt.savefig('./results/combined/cm_{:.3f}.jpeg'.format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--color_space', type=str, default='True', choices=['True', 'False'], help='whether to use color space classifier')
    parser.add_argument('--color_component', type=str, default='True', choices=['True', 'False'], help='whether to use color component classifier')
    parser.add_argument('--texture', type=str, default='True', choices=['True', 'False'], help='whether to use texture classifier')
    opt = parser.parse_args()

    test(data_path='./BoWFireDataset/dataset/img/', opt=opt)