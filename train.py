import os
from colorspace import ColorSpace_Classifier
from texture import Texture_Classfier
from component import ColorComponent_Classfier

def train_models(model_folder):
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    CS = ColorSpace_Classifier(n_neighbors=7)
    CS.train()
    CC = ColorComponent_Classfier(n_neighbors=9 ,method='default' , n_segments=100, max_bins=255, m=40)
    CC.train()
    TC = Texture_Classfier(n_neighbors=7, method='default', n_segments=120, max_bins=255, m=40)
    TC.train()

    print('\nTraining Done!')
    print(f'Models saved to {model_folder}.')

if __name__=='__main__':
    train_models(model_folder='./models/')