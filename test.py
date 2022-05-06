import os
import numpy as np
import torch, cv2
from CNN import CNN
import random
from skimage import io
from torchvision import transforms
from color import Color_Classifier
from texture import Texture_Classfier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

def test_CNN(model_path='./models/fire_0.96.pkl', data_path='./BoWFireDataset/dataset/img/'):
    model = CNN()
    model = torch.load(model_path)
    model.eval()
    test_images = os.listdir(data_path)
    if '.DS_Store' in test_images:
        test_images.remove('.DS_Store')
    random.shuffle(test_images)

    transform = transforms.ToTensor()

    total_correct = 0
    for img_name in test_images:
        img_path = os.path.join(data_path, img_name)
        img = cv2.imread(img_path)
        h,w,_ = img.shape
        img = img[0:int(h/50)*50:,0:int(w/50)*50,:]
        h,w,_ = img.shape
        fire_num = 0
        for i in range(0,int(h/50)):
            for j in range(0,int(w/50)):
                seg = img[50*i:50*(i+1),50*j:50*(j+1),:]
                seg = transform(seg).unsqueeze(0)

                predict_y = model(seg.float()).detach()
                predict_label = np.argmax(predict_y, axis=-1)

                if predict_label == 0:
                    fire_num += 1
        fire_rate = fire_num/int(h/50)/int(w/50)
        is_fire = (fire_rate > 0.03) and (fire_rate < 0.35)

        print(img_name, fire_num, int(h/50)*int(w/50),  is_fire)
        if img_name[0] == 'f' and is_fire:
            total_correct += 1
        elif img_name[0] == 'n' and (not is_fire):
            total_correct += 1

    acc = total_correct/len(test_images)
    print('accuracy: {:.2f}'.format(acc))
    
def test(data_path='./BoWFireDataset/dataset/img/', fire_threshold=0.005):
    CC = Color_Classifier()
    CC.train()
    TC = Texture_Classfier()
    TC.train()

    test_images = os.listdir(data_path)
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
        img = io.imread(img_path)
        h,w,_ = img.shape
        color_mask = CC.get_mask(img)
        texture_mask = TC.get_mask(img)
        mask = texture_mask * color_mask
        img_out = img * mask
        img_out_name = img_name[:-4] + '_out' + img_name[-4:] 
        img_save_path = os.path.join('./results/combined/', img_out_name)
        # io.imsave(img_save_path, img_out.astype(np.uint8))
        is_fire = mask.sum() > 300

        plt.figure(figsize=(10,8))
        plt.subplot(2,2,1)
        plt.imshow(img.astype(np.uint8))
        plt.title('Original Image')
        plt.subplot(2,2,2)
        plt.imshow((img * color_mask).astype(np.uint8))
        plt.title('Color Mask')
        plt.subplot(2,2,3)
        plt.imshow((img * texture_mask).astype(np.uint8))
        plt.title('Texture Mask')
        plt.subplot(2,2,4)
        plt.imshow(img_out.astype(np.uint8))
        plt.title('Fire Mask')
        plt.savefig(img_save_path)
        plt.close()

        truth.append(1 if is_fire else 0)
        predicted.append(1 if img_name[0] == 'f' else 0)
        print('[{}/{}]'.format(idx, len(test_images)) + img_name + ('\t fire' if is_fire else '\t no fire'))
    
    truth = np.array(truth)
    predicted = np.array(predicted)
    sns.set()
    f,ax = plt.subplots()
    cm = confusion_matrix(truth, predicted)
    cm = pd.DataFrame(cm,columns=["no fire", "fire"],index=["no fire", "fire"])
    sns.heatmap(cm,cmap="YlGnBu_r",fmt="d",annot=True)
    plt.show()
    
    acc = (truth==predicted).sum()/len(test_images)
    print('accuracy: {:.2f}'.format(acc))


if __name__ == "__main__":
    test()