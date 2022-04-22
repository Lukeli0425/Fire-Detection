import os
import torch
import numpy
import cv2

class BoWFireDataset():
    def __init__(self, data_path='./BoWFireDataset/train/', transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images = os.listdir(data_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.data_path, img_name)
        img = cv2.imread(img_path)

        if img_name[0] == 'f': # fire
            label = 0
        elif img_name[0] == 'n': # normal
            label = 1
        else: # smoke
            label = 2
        sample = {'image':img,'label':label}

        if self.transform:
            img = self.transform(img)
        return img, label

if __name__ == "__main__":
    print(BoWFireDataset().__getitem__(200)['label'])