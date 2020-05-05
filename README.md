![TF Depend](https://img.shields.io/badge/TensorFlow-2.1-orange) ![License Badge](https://img.shields.io/badge/license-Apache%202-green)<br>

<p align="center">
  <img width="150" src="./assets/logo.png">
</p>

<h2 align=center>Convolutional Nets in TensorFlow2</h2>

[ConvNets-TensorFlow2](https://github.com/marload/ConvNetsRL-TensorFlow2) is a repository that implements a variety of popular Deep Convolutional Network Architectures using [TensorFlow2](https://tensorflow.org). The core of this repository is intuitive code and concise architecture. If you are a user of TensorFlow2 and want to study various and popular CNN architectures, this repository will be the best choice to study. ConvNets-TensorFlow2 is continuously updated and managed. This repository has been very much influenced by [Cifar100-pytorch](https://github.com/weiaicunzai/pytorch-cifar100).

## Models

- [VGG](#vgg)
- [GoogLeNet](#googlenet)
- [ResNet](#resnet)
- [InceptionV3](#inceptionv3)
- [InceptionV4](#inceptionv4)
- [MobileNet](#mobilenet)
- [MobileNetV2](#mobilenetv2)
- [Squeezenet](#squeezenet)
- [SENet](#senet)


<hr>

<a name='vgg'></a>

### VGG

**Paper** [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)<br>
**Author** Karen Simonyan, Andrew Zissermanr<br>
**Code** [VGG.py](https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/VGG.py) 
<br><br>
**Model Options**
```bash
--nets {VGG11 or VGG13 or VGG16 or VGG19}
```

<hr>

<a name='googlenet'></a>

### GoogLeNet

**Paper** [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)<br>
**Author** Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich<br>
**Code** [GoogLeNet.py](https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/GoogLeNet.py)
<br><br>
**Model Options**
```bash
--nets {GoogLeNet}
```

<hr>

<a name='resnet'></a>

### ResNet

**Paper** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)<br>
**Author** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun<br>
**Code** [ResNet.py](https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/ResNet.py)
<br><br>
**Model Options**
```bash
--nets {ResNet18 or ResNet34 ResNet50 ResNet101 ResNet 152}
```

<hr>

<a name='inceptionv3'></a>

### InceptionV3

**Paper** [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)<br>
**Author** Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
<br>
**Code** [InceptionV3.py](https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/InceptionV3.py)
<br><br>
**Model Options**
```bash
--nets {InceptionV3}
```

<hr>

<a name='inceptionv4'></a>

### InceptionV4

**Paper** [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)<br>
**Author** Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
<br>
**Code** [InceptionV4.py](https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/InceptionV4.py)
<br><br>
**Model Options**
```bash
--nets {InceptionV4}
```

<hr>

<a name='mobilenet'></a>

### MobileNet

**Paper** [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)<br>
**Author** Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
<br>
**Code** [MobileNet.py](https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/MobileNet.py)
<br><br>
**Model Options**
```bash
--nets {MobileNet}
```

<hr>

<a name='mobilenetv2'></a>

### MobileNetV2

**Paper** [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)<br>
**Author** Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
<br>
**Code** [MobileNetV2.py](https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/MobileNetV2.py)
<br><br>
**Model Options**
```bash
--nets {MobileNetV2}
```

<hr>

<a name='squeezenet'></a>

### SqueezeNet

**Paper** [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)<br>
**Author** Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
<br>
**Code** [SqueezeNet.py](https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/SqueezeNet.py)
<br><br>
**Model Options**
```bash
--nets {SqueezeNet}
```

<hr>

<a name='SENet'></a>

### SENet

**Paper** [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)<br>
**Author** Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
<br>
**Code** [SEResNet.py](https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/SEResNet.py)
<br><br>
**Model Options**
```bash
--nets {SEResNet18 or SEResNet34 or SEResNet50 or SEResNet101 or SEResNet152}
```

<hr>








## Reference

- https://github.com/weiaicunzai/pytorch-cifar100
- https://github.com/tensorflow/tensorflow