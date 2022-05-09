# Fire-Detection


Superpixel：分割SLIC算法

https://blog.csdn.net/haoji007/article/details/103432879


SuperPiexle特征提取简介：可借鉴

https://blog.csdn.net/Resume_f/article/details/94180090

Superpixel：Skimage实现

https://blog.csdn.net/weixin_42990464/article/details/107753652?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.pc_relevant_antiscanv2&spm=1001.2101.3001.4242.1&utm_relevant_index=3


Superpixel特征提取:LBP特征

https://www.jianshu.com/p/039df67c2d5f

SVM训练：

https://www.jianshu.com/p/039df67c2d5f

训练：

`CNN-1`

`CNN-2`

`CNN-3`

用到的额外训练数据集:

第一级网络：Reviewer的训练

https://www.kaggle.com/datasets/phylake1337/fire-dataset


……

`CNN-n`

`Pixel-Texture-Net`

++ Adaboost集成



5月9日-5月13日：调试第一级CNN网络，第二级准备设计四五个小型LeNet CNN 分类器 + Color/Texture分类器---->得到90%以上的分类准确率即可

5月13日-5月15日：代码整合，在最后一级加入Color/Texture分割框架结果Mask，得到火焰所在位置Mask，整个框架输出01分类结果及Mask(0直接输出全黑图，1输出黑白图标记火焰位置)

5月15日-5月20日：代码整理，Readme文档编写，代码运行环境整理，预训练模型整理，数据集整理，Github代码整理，实验报告编写完成---->提交

5月21日：检查上述有无缺漏


`注：`代码实验中的可视化程序代码尽量不要整合时删除，争取保留下来，之后报告可以多放一些图

