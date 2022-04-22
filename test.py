import os
import numpy as np
import torch
import cv2
from CNN import CNN
import random
from torchvision import transforms

def test(model_path='./models/fire_0.96.pkl', data_path='./BoWFireDataset/dataset/img/'):
    model = CNN()
    model = torch.load(model_path)
    model.eval()
    test_images = os.listdir(data_path)
    test_images.remove('.DS_Store')
    random.shuffle( test_images)

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
    

if __name__ == "__main__":
    test()