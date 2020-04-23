import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model


class BottleNeck(layers.Layer):
    def __init__(self, growth_rate):
        super(BottleNeck, self).__init__()
        inner_channel = 4 * growth_rate

        self.bottle_neck = Sequential([
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(inner_channel, (1, 1), use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(growth_rate, (3, 3), padding='same', use_bias=False)
        ])

    def call(self, x, training=False):
        return tf.concat([x, self.bottle_neck(x, training=training)], axis=-1)


class Transition(layers.Layer):
    def __init__(self, out_channels):
        super(Transition, self).__init__()

        self.down_sample = Sequential([
            layers.BatchNormalization(),
            layers.Conv2D(out_channels, (1, 1), use_bias=False),
            layers.AveragePooling2D((2, 2), strides=2)
        ])

    def call(self, x, training=False):
        return self.down_sample(x, training=training)


class DenseNet(Model):
    def __init__(self,
                 num_classes,
                 block,
                 nblocks,
                 growth_rate=12,
                 reduction=0.5,
                 input_shape=(32, 32, 3)):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate

        self.conv1 = Sequential([
            layers.Input(input_shape),
            layers.Conv2D(inner_channels, (3, 3),
                          padding='same', use_bias=False)
        ])

        self.features = Sequential()

        for idx in range(len(nblocks) - 1):
            self.features.add(block(nblocks[idx]))
            inner_channels += growth_rate * nblocks[idx]

            out_channels = int(reduction * inner_channels)
            self.features.add(Transition(out_channels))
            inner_channels = out_channels

        self.features.add(self._make_dense_layers(
            block, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add(layers.BatchNormalization())
        self.features.add(layers.ReLU())

        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def _make_dense_layers(self, block, nblocks):
        dense_block = Sequential()
        for idx in range(nblocks):
            dense_block.add(block(self.growth_rate))
        return dense_block

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.features(x, training=training)
        x = self.gap(x)
        x = self.fc(x)
        return x


def densenet121(num_classes):
    return DenseNet(num_classes, BottleNeck, [6, 12, 24, 16], growth_rate=32)


def densenet169(num_classes):
    return DenseNet(num_classes, BottleNeck, [6, 12, 32, 32], growth_rate=32)


def densenet201(num_classes):
    return DenseNet(num_classes, BottleNeck, [6, 12, 48, 32], growth_rate=32)


def densenet161(num_classes):
    return DenseNet(num_classes, BottleNeck, [6, 12, 36, 24], growth_rate=48)
