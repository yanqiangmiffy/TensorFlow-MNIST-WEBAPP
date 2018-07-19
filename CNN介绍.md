## 1 CNN的前生今世
### 1.1 大脑
作为人类，我们不断地通过眼睛来观察和分析周围的世界，我们不需要刻意的“努力”思考，就可以对岁看到的一切做出预测，并对它们采取行动。当我们看到某些东西时，我们会根据我们过去学到的东西来标记每个对象。为了说明这些情况，请看下面这张图片：
![资料来源：https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050](https://upload-images.jianshu.io/upload_images/1531909-8c047d8884aa6455.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
你可能会想到“这是一个快乐的小男孩站在椅子上”。或者也许你认为他看起来像是在尖叫，即将在他面前攻击这块蛋糕。

这就是我们整天下意识地做的事情。我们看到事物，然后标记，进行预测和识别行为。但是我们怎么做到这些的呢？我们怎么能解释我们看到的一切？

大自然花费了5亿多年的时间来创建一个系统来实现这一目标。眼睛和大脑之间的合作，称为主要视觉通路，是我们理解周围世界的原因。
![视觉通路。- 来源：https：//commons.wikimedia.org/wiki/File : Human_visual_pathway.svg
](https://upload-images.jianshu.io/upload_images/1531909-985ecf9d00bee373.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
虽然视力从眼睛开始，但我们所看到的实际解释发生在大脑的**初级视觉皮层中**。

当您看到一个物体时，您眼中的光感受器会通过视神经将信号发送到正在处理输入的主视觉皮层。在[初级视觉皮层](https://www.youtube.com/watch?v=unWnZvXJH2o&t=516s)，使眼睛看到的东西的感觉。

所有这一切对我们来说都很自然。我们几乎没有想到我们能够识别我们生活中看到的所有物体和人物的特殊性。神经元和大脑连接的**深层复杂层次结构**在记忆和标记物体的过程中起着重要作用。

想想我们如何学习例如伞是什么。或鸭子，灯，蜡烛或书。一开始，我们的父母或家人告诉我们直接环境中物体的名称。我们通过给我们的例子了解到。慢慢地，但我们开始在我们的环境中越来越多地认识到某些事情。它们变得如此普遍，以至于下次我们看到它们时，我们会立即知道这个物体的名称是什么。他们成为我们世界的**模型**一部分。
### 1.2 卷积神经网络的历史
与孩子学会识别物体的方式类似，我们需要在能够概括输入并对之前从未见过的图像进行预测之前，展示数百万张图片的算法。

计算机以与我们不同的方式“看到”东西的。他们的世界只包括数字。每个图像都可以表示为二维数字数组，称为像素。

但是它们以不同的方式感知图像，这一事实并不意味着我们无法训练他们的识别模式，就像我们一样如何识别图像。我们只需要以不同的方式思考图像是什么。
![计算机如何看到图像。- 来源：http：//cs231n.github.io/classification/
](https://upload-images.jianshu.io/upload_images/1531909-31d1b48636edec48.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
为了“教会”一种算法如何识别图像中的对象，我们使用特定类型的[人工神经网络](https://medium.com/@daphn3cor/building-a-3-layer-neural-network-from-scratch-99239c4af5d3)：卷积神经网络（CNN）。他们的名字源于网络中最重要的一个操作：[卷积](https://en.wikipedia.org/wiki/Convolution)。

卷积神经网络受到大脑的启发。DH Hubel和TN Wiesel在20世纪50年代和60年代对哺乳动物大脑的研究提出了哺乳动物如何在视觉上感知世界的新模型。他们表明猫和猴的视觉皮层包括在其直接环境中专门响应神经元的神经元。

在他们的[论文中](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1359523/pdf/jphysiol01247-0121.pdf)，他们描述了大脑中两种基本类型的视觉神经元细胞，每种细胞以不同的方式起作用：简单细胞（**S细胞**）和复合细胞（**C细胞**）。

例如，当简单单元格将基本形状识别为固定区域和特定角度的线条时，它们就会激活。复杂细胞具有较大的感受野，其输出对野外的特定位置不敏感。

复杂细胞继续对某种刺激做出反应，即使它在[视网膜](https://en.wikipedia.org/wiki/Retina)上的绝对位置发生变化。在这种情况下，复杂指的是更灵活。

在[视觉中](http://www.cns.nyu.edu/~david/courses/perception/lecturenotes/V1/lgn-V1.html)，单个感觉神经元的**感受**区域是视网膜的特定区域，其中某些东西将影响该神经元的发射（即，将激活神经元）。每个感觉神经元细胞都有相似的感受野，它们的田地覆盖着。
![神经元的感受野。- 来源：http：//neuroclusterbrain.com/neuron_model.html](https://upload-images.jianshu.io/upload_images/1531909-7eda90096de63400.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

此外，**层级【hierarchy 】**的概念在大脑中起着重要作用。信息按顺序存储在模式序列中。的**新皮层**，它是大脑的最外层，以分层方式存储信息。它存储在皮质柱中，或者在新皮层中均匀组织的神经元分组。

1980年，一位名为Fukushima的研究员提出了一种[分层神经网络模型](https://en.wikipedia.org/wiki/Neocognitron)。他称之为新**认知**。该模型的灵感来自简单和复杂细胞的概念。neocognitron能够通过了解物体的形状来识别模式。

后来，1998年，卷心神经网络被Bengio，Le Cun，Bottou和Haffner引入。他们的第一个卷积神经网络称为**LeNet-5**，能够对手写数字中的数字进行分类。
![LeNet-5网络 示意图1](https://upload-images.jianshu.io/upload_images/1531909-a743c7747d48b0f2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![LeNet-5网络 示意图2](https://upload-images.jianshu.io/upload_images/1531909-2f884eb5cbf1a7a2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 2 卷积神经网络
卷积神经网络（Convolutional Neural Network）简称CNN，CNN是所有深度学习课程、书籍必教的模型，CNN在影像识别方面的为例特别强大，许多影像识别的模型也都是以CNN的架构为基础去做延伸。另外值得一提的是CNN模型也是少数参考人的大脑视觉组织来建立的深度学习模型，学会CNN之后，对于学习其他深度学习的模型也很有帮助，本文主要讲述了CNN的原理以及使用CNN来达成99%正确度的手写字体识别。
CNN的概念图如下：
![CNN概念图1](https://upload-images.jianshu.io/upload_images/1531909-563b7aaa812d960d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![CNN概念图2](https://upload-images.jianshu.io/upload_images/1531909-8d69ee9ee863756b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![CNN概念图3](https://upload-images.jianshu.io/upload_images/1531909-e0cedbb4d09b3d1c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
从上面三张图片我们可以看出，CNN架构简单来说就是：图片经过各两次的Convolution, Pooling, Fully Connected就是CNN的架构了，因此只要搞懂Convolution, Pooling, Fully Connected三个部分的内容就可以完全掌握了CNN！
### 2.1  Convolution Layer卷积层
卷积运算就是将原始图片的与特定的`Feature Detector(filter)`做卷积运算(符号`⊗`)，卷积运算就是将下图两个`3x3`的矩阵作相乘后再相加，以下图为例`0 *0 + 0*0 + 0*1+ 0*1 + 1 *0 + 0*0 + 0*0 + 0*1 + 0*1 =0`
![卷积运算 1](https://upload-images.jianshu.io/upload_images/1531909-279c4912744eca3f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
每次移动一步，我们可以一次做完整张表的计算，如下：
![卷积运算 2](https://upload-images.jianshu.io/upload_images/1531909-8ed1594933d1bcbc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
下面的动图更好地解释了计算过程：
![左：过滤器在输入上滑动。右：结果汇总并添加到要素图中。](https://upload-images.jianshu.io/upload_images/1531909-9bdd5096b12aa5fd.gif?imageMogr2/auto-orient/strip)

中间的Feature Detector(Filter)会随机产生好几种(ex:16种)，Feature Detector的目的就是帮助我们萃取出图片当中的一些特征(ex:形状)，就像人的大脑在判断这个图片是什么东西也是根据形状来推测
![16种不同的Feature Detector](https://upload-images.jianshu.io/upload_images/1531909-df086e2364853b15.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
利用Feature Detector萃取出物体的边界
![利用Feature Detector萃取出物体的边界](https://upload-images.jianshu.io/upload_images/1531909-30902a3ec368c64f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
使用Relu函数去掉负值，更能淬炼出物体的形状
![Relu函数去掉负值](https://upload-images.jianshu.io/upload_images/1531909-c97f50737816652a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![淬炼出物体的形状1](https://upload-images.jianshu.io/upload_images/1531909-27526811f5eb4d99.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![淬炼出物体的形状2](https://upload-images.jianshu.io/upload_images/1531909-439b4790b64ae66e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们在输入上进行了多次卷积，其中每个操作使用不同的过滤器。这导致不同的特征映射。最后，我们将所有这些特征图放在一起，作为卷积层的最终输出。

就像任何其他神经网络一样，我们使用**激活函数**使输出非线性。在卷积神经网络的情况下，卷积的输出将通过激活函数。这可能是[ReLU](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions)激活功能

![其他函数](https://upload-images.jianshu.io/upload_images/1531909-4063759e53c2bc5c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这里还有一个概念就是**步长**，**Stride**是每次卷积滤波器移动的步长。步幅大小通常为1，意味着滤镜逐个像素地滑动。通过增加步幅大小，您的滤波器在输入上滑动的间隔更大，因此单元之间的重叠更少。

下面的动画显示步幅大小为1。
![步幅为1](https://upload-images.jianshu.io/upload_images/1531909-659bf8ba260aa833.gif?imageMogr2/auto-orient/strip)
由于feature map的大小始终小于输入，我们必须做一些事情来防止我们的要素图缩小。这是我们使用填充的地方。

添加一层零值像素以使用零环绕输入，这样我们的要素图就不会缩小。除了在执行卷积后保持空间大小不变，填充还可以提高性能并确保内核和步幅大小适合输入。

可视化卷积层的一种好方法如下所示，最后我们以一张动图解释下卷积层到底做了什么
![卷积如何与K = 2滤波器一起工作，每个滤波器具有空间范围F = 3，步幅S = 2和输入填充P = 1. - 来源：http：//cs231n.github.io/convolutional-networks/](https://upload-images.jianshu.io/upload_images/1531909-36da4908829470ff.gif?imageMogr2/auto-orient/strip)

### 2.2 Pooling Layer 池化层
在卷积层之后，通常在CNN层之间添加**池化层**。池化的功能是不断降低维数，以减少网络中的参数和计算次数。这缩短了训练时间并控制[过度拟合](https://en.wikipedia.org/wiki/Overfitting)。

最常见的池类型是**max pooling**，它在每个窗口中占用最大值。需要事先指定这些窗口大小。这会降低特征图的大小，同时保留重要信息。
>Max Pooling主要的好处是当图片整个平移几个Pixel的话对判断上完全不会造成影响，以及有很好的抗杂讯功能。

![池化层 示意图 1](https://upload-images.jianshu.io/upload_images/1531909-f3daf861d882404f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![池化层 示意图 2](https://upload-images.jianshu.io/upload_images/1531909-2d6fb7eed9d3c857.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 2.3 Fully Connected Layer 全连接层
基本上全连接层的部分就是将之前的结果平坦化之后接到最基本的神经网络了
![](https://upload-images.jianshu.io/upload_images/1531909-20b592fd47809153.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![](https://upload-images.jianshu.io/upload_images/1531909-4ecdc2dfefb56f54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![](https://upload-images.jianshu.io/upload_images/1531909-4bc198275647d005.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 3 利用CNN识别MNIST手写字体
下面这部分主要是关于如歌使用tensorflow实现CNN以及手写字体识别的应用
```
# CNN 代码
def convolutional(x,keep_prob):

    def conv2d(x,W):
        return tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    def weight_variable(shape):
        initial=tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    x_image=tf.reshape(x,[-1,28,28,1])
    W_conv1=weight_variable([5,5,1,32])
    b_conv1=bias_variable([32])
    h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1=max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # full_connetion
    W_fc1=weight_variable([7*7*64,1024])
    b_fc1=bias_variable([1024])
    h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

    # dropout 随机扔掉一些值，防止过拟合
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

    W_fc2=weight_variable([1024,10])
    b_fc2=bias_variable([10])
    y=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

    return y,[W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2]
```
大家稍微对tensorflow的代码有些基础，理解上面这部分基本上没有难度，并且基本也是按照我们前面概念图中的逻辑顺序实现的。

最终按照慕课网上的学习资料[TensorFlow与Flask结合打造手写体数字识别](https://www.imooc.com/learn/994)，实现了一遍CNN,比较曲折的地方是前端，以及如何将训练的模型与flask整合，最后项目效果如下：
![来源 https://github.com/yanqiangmiffy/TensorFlow-MNIST-WEBAPP](https://upload-images.jianshu.io/upload_images/1531909-af303ba39812f0cf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
欢迎大家到GitHub fork和star,项目传送门--->[TensorFlow-MNIST-WEBAPP](https://github.com/yanqiangmiffy/TensorFlow-MNIST-WEBAPP)

## 4 总结
最后说自己的两点感触吧：
- CNN在各种场景已经应用很成熟，网上资料特别多，原先自己也是略知一二，但是从来没有总结整理过，还是整理完之后心里比较踏实一些。
- 切记理论加实践，实现一遍更踏实。

## 5 参考资料

- [[資料分析&機器學習] 第5.1講: 卷積神經網絡介紹(Convolutional Neural Network) ](https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC5-1%E8%AC%9B-%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E7%B5%A1%E4%BB%8B%E7%B4%B9-convolutional-neural-network-4f8249d65d4f)
- [An Intuitive Explanation of Convolutional Neural Networks – the data science blog ](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
- [Convolutional Neural Network (CNN) | Skymind ](https://skymind.ai/wiki/convolutional-network)
- [Convolutional Neural Networks (LeNet) — DeepLearning 0.1 documentation ](http://deeplearning.net/tutorial/lenet.html)
- [CS231n Convolutional Neural Networks for Visual Recognition ](http://cs231n.github.io/convolutional-networks/)
- [卷积神经网络(CNN)学习笔记1：基础入门 | Jey Zhang ](http://www.jeyzhang.com/cnn-learning-notes-1.html)
- [Deep Learning（深度学习）学习笔记整理系列之（七） - CSDN博客 ](https://blog.csdn.net/zouxy09/article/details/8781543)
