import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model


def ReLU6():
    return layers.Lambda(lambda x: tf.nn.relu6(x))


class LinearBottleNeck(layers.Layer):
    def __init__(self, in_channels, out_channels, strides=1, t=6):
        super(LinearBottleNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides

        self.residual = Sequential([
            layers.Conv2D(in_channels * t,
                          (1, 1),
                          strides=1,
                          padding='same'),
            layers.BatchNormalization(),
            ReLU6(),
            layers.DepthwiseConv2D((3, 3),
                                   strides=strides,
                                   padding='same'),
            layers.BatchNormalization(),
            ReLU6(),
            layers.Conv2D(out_channels,
                          (1, 1),
                          strides=1,
                          padding='same'),
            layers.BatchNormalization(),
        ])

    def call(self, x, training=False):
        residual = self.residual(x, training=training)

        if self.strides == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2(Model):
    def __init__(self, num_classes, input_shape=(32, 32, 3)):
        super(MobileNetV2, self).__init__()

        self.front = Sequential([
            layers.Input(input_shape),
            layers.BatchNormalization(),
            ReLU6()
        ])
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = layers.Conv2D(filters=1280,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation='softmax')
    def _make_stage(self, repeat, in_channels, out_channels, strides, t):
        nets = Sequential()
        nets.add(LinearBottleNeck(in_channels, out_channels, strides, t))

        while repeat - 1:
            nets.add(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1
        return nets

    def call(self, inputs, training=False):
        x = self.front(inputs)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


def mobilenetv2(num_classes):
    return MobileNetV2(num_classes)
