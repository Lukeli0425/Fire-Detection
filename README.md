# Fire-Detection

This is a _non-deep learning_ **fire detection** pipeline inspired by [_this paper_](https://ieeexplore.ieee.org/abstract/document/7314551). The pipeline can be divided into three parts: **color space classifier**, **color component classifier** and t**exture classifier**. Our model was trained and tested on the [__BoWFire Dataset__](https://bitbucket.org/gbdi/bowfire-dataset/src/master/).

[![OxD1l6.png](https://s1.ax1x.com/2022/05/22/OxD1l6.png)](https://imgtu.com/i/OxD1l6)

## Environment Setup

Install the required packages for this repo:

```zsh
pip install -r requirements.txt
```

## Train the Models

Train the models with
```zsh
python train.py
```
The trained models will be saved under the 'models' folder.

## Test the Models

Test the models on the BoWFire Dataset with

```zsh
python test.py
```

The default setting is to use all three classifiers together. The results will be saved under the 'results' folder. If you want to use certain classifiers, run:

```zsh
python test.py  --color_space [True/False]  --color_component [True/False]  --texture [True/False]
```

## File list

```bash
...
+ BoWFireDataset              # train and test Dataset
+ models                      # models to be saved
+ references                  # reference paper
+ results                     # experiment results
README.md 
requirements.txt              # environment prerequisites
colorspace.py                 # color space classifier
component.py                  # color component classifier
texture.py                    # texture classifier
train.py                      # train the model
test.py                       # test the model
run.sh                        # train and test the model together
```

