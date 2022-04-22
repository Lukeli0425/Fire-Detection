***********************
* Dataset description *
***********************
This dataset consists of two sets of images:
    - training (dataset.csv)
    - test (trainset.csv)

The training dataset (./train) consists of 240 images of 50×50 pixels
resolution; 80 images classified as fire, and 160 as non-fire. Note 
that the non-fire images also contain red or yellow objects.

The test dataset (./dataset) consists of 226 images with various 
resolutions. It is divided in two categories:
    - 119 images containing fire
    - 107 images without fire.
The fire images consist of emergency situations with different fire 
incidents, as buildings on fire, industrial fire, car accidents, and 
riots (./dataset/img). The remaining images consist of emergency 
situations with no visible fire and also images with fire-like 
regions, such as sunsets, and red or yellow objects. These images 
were manually cropped by human experts to create a ground-truth of 
fire regions. The fire regions on the ground-truth images are marked 
in white while non-fire regions are marked with black (./dataset/gt).

**************
* Created by *
**************
Daniel Y. T. Chino and Letricia P. S. Avalhais

*****************
* Data citation *
*****************
When publishing researched based on this dataset, please include a citation for the dataset as following:
    D. Y. T. Chino, L. P. S. Avalhais, J. F. Rodrigues and A. J. M. Traina, "BoWFire: Detection of Fire in Still Images by Integrating Pixel Color and Texture Analysis," 2015 28th SIBGRAPI Conference on Graphics, Patterns and Images, Salvador, 2015, pp. 95-102.
    [doi: 10.1109/SIBGRAPI.2015.19]

************
* Licenses *
************
All images were collect from Flickr under the creative commons license.