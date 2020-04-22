import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model


class BasicBlock(layers.Layer):
    def __init__(self, kernels, stride=1):
        super(BasicBlock, self).__init__()

        self.features = Sequential([
            layers.Conv2D(kernels, (3, 3), strides=stride, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(kernels, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization()
        ])

        
        if stride != 1:
            shortcut = [
                layers.Conv2D(kernels, (1, 1), strides=stride),
                layers.BatchNormalization()
            ]
        else:
            shortcut = []
        self.shorcut = Sequential(shortcut)

    def call(self, inputs, training=False):
        residual = self.shorcut(inputs, training=training)
        x = self.features(inputs, training=training)
        x = tf.nn.relu(layers.add([residual, x]))
        return x


class BottleNeckBlock(layers.Layer):
    def __init__(self, kernels, stride=1):
        super(BottleNeckBlock, self).__init__()

        self.features = Sequential([
            layers.Conv2D(kernels, (1, 1), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(kernels, (3, 3), strides=stride, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(kernels * 4, (1, 1), strides=1, padding='same'),
            layers.BatchNormalization(),
        ])

        self.shorcut = Sequential([
            layers.Conv2D(kernels * 4, (1, 1), strides=stride),
            layers.BatchNormalization()
        ])

    def call(self, inputs, training=False):
        residual = self.shorcut(inputs, training=training)
        x = self.features(inputs, training=training)
        x = tf.nn.relu(x + residual)
        return x


class ResNet(Model):
    def __init__(self, block, num_blocks, num_classes, input_shape=(32, 32, 3)):
        super(ResNet, self).__init__()
        self.conv1 = Sequential([
            layers.Input(input_shape),
            layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def _make_layer(self, block, kernels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        nets = []
        for stride in strides:
            nets.append(block(kernels, stride))
        return Sequential(nets)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes):
    return ResNet(BottleNeckBlock, [3, 4, 6, 3], num_classes)


def resnet101(num_classes):
    return ResNet(BottleNeckBlock, [3, 4, 23, 3], num_classes)


def resnet152(num_classes):
    return ResNet(BottleNeckBlock, [3, 8, 36, 3], num_classes)
