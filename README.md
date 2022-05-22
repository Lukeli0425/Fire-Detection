# Fire-Detection

This is a non-deep learning fire detection pipeline inspired by this [paper](https://ieeexplore.ieee.org/abstract/document/7314551). The pipeline can be divided into three parts: color space classifier, color component classifier and texture classifier. Our model was trained and tested on the [BoWFire Dataset](https://bitbucket.org/gbdi/bowfire-dataset/src/master/).

## Environment Setup
Install the required packages:
```zsh
pip install -r requirements.txt
```

## Run the code
To train and test the model on the BoWFire Dataset:
```zsh
python test.py
```
The default settings is to use all three classifiers. The results will be saved in ./results/. If you want to use choose certain classifiers, run
```zsh
python test.py  --color_space False  --color_component True  --texture True
```
