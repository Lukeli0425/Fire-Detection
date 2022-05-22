# Fire-Detection

This is a _non-deep learning_ **fire detection** pipeline inspired by [_this paper_](https://ieeexplore.ieee.org/abstract/document/7314551). The pipeline can be divided into three parts: **color space classifier**, **color component classifier** and **texture classifier**. Our models were trained and tested on the [_BoWFire Dataset_](https://bitbucket.org/gbdi/bowfire-dataset/src/master/).

## Pipline

[![OxD1l6.png](https://s1.ax1x.com/2022/05/22/OxD1l6.png)](https://imgtu.com/i/OxD1l6)

## Environment Setup

Clone the code:

```zsh
git clone https://github.com/Lukeli0425/Fire-Detection.git
```

Install the required packages for this repo:

```zsh
pip install -r requirements.txt
```

## Train the Models

Train the models with

```zsh
python train.py
```

The trained models will be saved under the `models` folder.

## Test the Models

Test the models on the BoWFire Dataset with

```zsh
python test.py
```

The default setting is to use all three classifiers together. The results will be saved under the `results`folder. If you want to use certain classifiers, run:

```zsh
python test.py  --color_space [True/False]  --color_component [True/False]  --texture [True/False]
```



## Train and Test together

```zsh
chmod +x run.sh
./run.sh
```

### Best Results

[![OxjLaF.jpg](https://s1.ax1x.com/2022/05/22/OxjLaF.jpg)](https://imgtu.com/i/OxjLaF)

## File list

```bash
...
+ BoWFireDataset              # train & test dataset
+ models                      # saved models
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
