# Convolution Neural Network and its Application in Image Denoising

## Reference
  1. https://arxiv.org/pdf/1608.03981.pdf
  2. https://arxiv.org/pdf/1608.03981v1.pdf
  3. http://www.skoltech.ru/app/data/uploads/sites/19/2017/06/1320.pdf

## Code in Python
  1. https://github.com/apar-singhal/CNN-Image-Denoising
  2. https://github.com/cszn/DnCNN
  3. https://github.com/cig-skoltech/NLNet

## A Survey of Deep Learning and Image Denoising
  1. 2015
    深度学习、自编码器、低照度图像增强
    Lore, Kin Gwn, Adedotun Akintayo, and Soumik Sarkar. "LLNet: A Deep Autoencoder Approach to Natural Low-light Image Enhancement." arXiv preprint arXiv:1511.03995 (2015).
    利用深度学习的自编码器方法训练不同低照度图像信号的特征来实现自适应变亮和去噪，主要是通过非线性暗化和添加高斯噪声的方法来模拟低照度环境，进行图像对比度增强和去噪。
    
    
  2. 2014
    深度学习、深度卷积神经网络、图像去卷积
    Xu, Li, et al. "Deep convolutional neural network for image deconvolution."Advances in Neural Information Processing Systems. 2014.
    利用深度卷积神经网络进行图像去卷积，实现图像复原，优点：相比于当前其他方法，有更好的PSNR值和视觉效果。
    
    
  3. 2014
    深度学习、稀疏编码、自编码器、图像去噪
    Li, HuiMing. "Deep Learning for Image Denoising." International Journal of Signal Processing, Image Processing and Pattern Recognition 7.3 (2014): 171-180.
    利用稀疏编码（sparsecoding）与自编码器（Auto-encoder）两种方法结合来实现图像去噪，不足之处是只对图像进行处理，没有涉及视频。
    
    
  4. 2014
    深度学习、rectified linear函数、深度神经网络、图像去噪
    Wu, Yangwei, Haohua Zhao, and Liqing Zhang. "Image Denoising with Rectified Linear Units." Neural Information Processing. Springer International Publishing, 2014.
    利用rectified linear (Re L) 函数代替sigmoid 函数作为深度神经网络的隐藏层的激活函数，来实现图像去噪；利用随机梯度下降的方法训练含噪图像和无噪图像来估计神经网络的参数；优点：和sigmoid函数作为激活函数的深度神经网络相比，能得到更好的去噪效果和更快的收敛速度。
    
    
  5. 2013
    深度学习、堆叠式稀疏去噪自编码器SSDAs、深度神经网络DNN、图像去噪
    Agostinelli, Forest, Michael R. Anderson, and Honglak Lee. "Robust image denoising with multi-column deep neural networks." Advances in Neural Information Processing Systems. 2013.
    利用改进的堆叠式稀疏去噪自编码器(Stacked sparse denoising autoencoders (SSDAs))，通过组合多个SSDAs，求解一个非线性优化方程计算每个SSDAs的最优权重，同时训练单独的网络去预测最优权重，实现视频去噪。优点：解决了SSDAs只能处理训练集中出现的噪声，这种方法可以处理训练集未出现的噪声类型。
    
    
  6. 2012
    深度学习、多层感知器、图像去噪
    Burger, Harold C., Christian J. Schuler, and Stefan Harmeling. "Image denoising: Can plain Neural Networks compete with BM3D?." Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.
    利用普通的多层感知器plain multi layer perceptron(MLP)实现图像去噪。
    
    
  7. 2012
    深度学习、稀疏编码、去噪自编码器、图像去噪
    Xie, Junyuan, Linli Xu, and Enhong Chen. "Image denoising and  inpainting with deep neural networks." Advances in Neural Information Processing Systems. 2012.
    将稀疏编码（sparse coding）与去噪自编码器(denoising auto-encoders)预训练的深度神经网络相结合进行图像去噪，噪声类型：高斯白噪声的灰度图像，但是稍微扩展下也可以处理彩色图，优点：比线性稀疏编码去除高斯白噪声的效果要好，不足之处：非常依赖有监督的训练，只能除去训练集中出现的噪声。
    
    
  8. 2012
    深度学习、多层感知器、图像去噪
    Burger, Harold Christopher, Christian J. Schuler, and Stefan Harmeling. "Image denoising with multi-layer perceptrons, part 1: comparison with existing algorithms and with bounds." arXiv preprint arXiv:1211.1544 (2012).
    利用多层感知器Multi-layer perceptions(MLP) 的方法实现图像去噪，噪声类型：This approach is easily adapted to less extensively studied types of noise, such as mixed Poisson-Gaussian noise, JPEG artifacts, salt-and-pepper noise and noise resembling stripes.
    
    
  9. 2010
    深度学习、堆叠式去噪自编码器、图像去噪
    Vincent, Pascal, et al. "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion." The Journal of Machine Learning Research 11 (2010): 3371-3408.
    利用堆叠式去噪自编码器（SDA）的方法进行图像去噪，堆叠式自编码器这种方法是深度学习中构建深度架构的重要方法之一。当训练出一个自编码器后，就可以在此基础上通过将第一个自编码器的输出作为第二个自编码器的输入继续训练出一个新的自编码器。这样继续训练下去就可以得到一个多层的堆叠式自编码器（Stacked  Autoencoders）。
    
    
  10. 2009
    深度学习、卷积网络、图像去噪
    Jain, Viren, and Sebastian Seung. "Natural image denoising with convolutional networks." Advances in Neural Information Processing Systems. 2009.
    利用卷积网络实现自然图像去噪。

    ————————————————
    版权声明：本文为CSDN博主「zhihua_bupt」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
    [原文链接](https://blog.csdn.net/geekmanong/article/details/50572148)
    
