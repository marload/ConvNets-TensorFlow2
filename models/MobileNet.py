import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model


class MobileNet(Model):
    def __init__(self, num_classes, alpha=1, input_shape=(32, 32, 3)):
        super(MobileNet, self).__init__()

        self.conv1 = Sequential([
            layers.Input(input_shape),
            layers.Conv2D(32,
                          (3, 3),
                          strides=2,
                          padding='same',
                          activation='relu'),
            layers.SeparableConv2D(64,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
        ])
        self.conv2 = Sequential([
            layers.SeparableConv2D(128,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
            layers.SeparableConv2D(128,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
        ])
        self.conv3 = Sequential([
            layers.SeparableConv2D(1256,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
            layers.SeparableConv2D(256,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
        ])
        self.conv4 = Sequential([
            layers.SeparableConv2D(512,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
            layers.SeparableConv2D(512,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
            layers.SeparableConv2D(512,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
            layers.SeparableConv2D(512,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
            layers.SeparableConv2D(512,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
            layers.SeparableConv2D(512,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
        ])
        self.conv5 = Sequential([
            layers.SeparableConv2D(1024,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
            layers.SeparableConv2D(1024,
                                   (3, 3),
                                   strides=1,
                                   padding="same",
                                   depth_multiplier=alpha,
                                   activation='relu'),
        ])
        self.ap = layers.AveragePooling2D((7, 7), strides=1)
        self.flat = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.ap(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


def mobilenet(num_classes):
    return MobileNet(num_classes)
