import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(Model):
    def __init__(self, features, num_classes, input_shape=(32, 32, 3)):
        super(VGG, self).__init__()
        
        self.features = Sequential([
            layers.Input(input_shape),
            features
        ])

        self.classifier = Sequential([
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax'),
        ])

    def call(self, inputs, training=False):
        x = self.features(inputs, training=training)
        x = self.classifier(x, training=training)
        return x


def make_layers(cfg):
    nets = []

    for l in cfg:
        if l == 'M':
            nets += [layers.MaxPool2D()]
            continue

        nets += [layers.Conv2D(l, (3, 3), padding='same')]
        nets += [layers.BatchNormalization()]
        nets += [layers.ReLU()]
    return Sequential(nets)


def vgg11(num_classes):
    return VGG(make_layers(cfg['A']), num_classes)


def vgg13(num_classes):
    return VGG(make_layers(cfg['B']), num_classes)


def vgg16(num_classes):
    return VGG(make_layers(cfg['D']), num_classes)


def vgg19(num_classes):
    return VGG(make_layers(cfg['E']), num_classes)
