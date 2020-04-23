import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model


class BasicConv2D(layers.Layer):
    def __init__(self, kernels, kernel_size=(3, 3), strides=1, padding='valid'):
        super(BasicConv2D, self).__init__(self)
        self.conv = layers.Conv2D(kernels,
                                  kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  use_bias=False)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class InceptionA(layers.Layer):
    def __init__(self, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2D(64, (1, 1))
        self.branch5x5 = Sequential([
            BasicConv2D(48, (1, 1)),
            BasicConv2D(64, (5, 5), padding='same')
        ])
        self.branch3x3 = Sequential([
            BasicConv2D(64, (1, 1)),
            BasicConv2D(96, (3, 3), padding='same'),
            BasicConv2D(96, (3, 3), padding='same')
        ])
        self.branchpool = Sequential([
            layers.AveragePooling2D((3, 3), strides=1, padding='same'),
            BasicConv2D(pool_features, (3, 3), padding='same')
        ])

    def call(self, x, training=False):
        branch1x1 = self.branch1x1(x, training=training)
        branch5x5 = self.branch5x5(x, training=training)
        branch3x3 = self.branch3x3(x, training=training)
        branchpool = self.branchpool(x, training=training)
        outputs = [branch1x1, branch5x5, branch3x3, branchpool]
        return tf.concat(outputs, axis=-1)  # TODO CHECK AXIS


class InceptionB(layers.Layer):
    def __init__(self):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2D(384, (3, 3), strides=2)
        self.branch3x3stack = Sequential([
            BasicConv2D(64, (1, 1)),
            BasicConv2D(96, (3, 3), padding='same'),
            BasicConv2D(96, (3, 3), strides=2)
        ])
        self.branchpool = layers.MaxPooling2D((3, 3), strides=2)

    def call(self, x, training=False):
        branch3x3 = self.branch3x3(x, training=training)
        branch3x3stack = self.branch3x3stack(x, training=training)
        branchpool = self.branchpool(x, training=training)
        outputs = [branch3x3, branch3x3stack, branchpool]
        return tf.concat(outputs, axis=-1)


class InceptionC(layers.Layer):
    def __init__(self, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2D(192, (1, 1))
        c7 = channels_7x7
        self.branch7x7 = Sequential([
            BasicConv2D(c7, (1, 1)),
            layers.ZeroPadding2D((3, 0)),
            BasicConv2D(c7, (7, 1)),
            layers.ZeroPadding2D((0, 3)),
            BasicConv2D(192, (1, 7))
        ])
        self.branch7x7stack = Sequential([
            BasicConv2D(c7, (1, 1)),
            layers.ZeroPadding2D((3, 0)),
            BasicConv2D(c7, (7, 1)),
            layers.ZeroPadding2D((0, 3)),
            BasicConv2D(c7, (1, 7)),
            layers.ZeroPadding2D((3, 0)),
            BasicConv2D(c7, (7, 1)),
            layers.ZeroPadding2D((0, 3)),
            BasicConv2D(192, (1, 7)),
        ])
        self.branchpool = Sequential([
            layers.AveragePooling2D((3, 3), strides=1, padding='same'),
            BasicConv2D(192, (1, 1))
        ])

    def call(self, x, training=False):
        branch1x1 = self.branch1x1(x, training=training)
        branch7x7 = self.branch7x7(x, training=training)
        branch7x7stack = self.branch7x7stack(x, training=training)
        branchpool = self.branchpool(x, training=training)
        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]
        return tf.concat(outputs, 3)


class InceptionD(layers.Layer):
    def __init__(self):
        super(InceptionD, self).__init__()
        self.branch3x3 = Sequential([
            BasicConv2D(192, (1, 1)),
            BasicConv2D(320, (3, 3), strides=2)
        ])
        self.branch7x7 = Sequential([
            BasicConv2D(192, (1, 1)),
            layers.ZeroPadding2D((0, 3)),
            BasicConv2D(192, (1, 7)),
            layers.ZeroPadding2D((3, 0)),
            BasicConv2D(192, (7, 1)),
            BasicConv2D(192, (3, 3), strides=2)
        ])
        self.branchpool = layers.AveragePooling2D((3, 3), strides=2)

    def call(self, x, training=False):
        branch3x3 = self.branch3x3(x, training=training)
        branch7x7 = self.branch7x7(x, training=training)
        branchpool = self.branchpool(x, training=training)
        outputs = [branch3x3, branch7x7, branchpool]
        return tf.concat(outputs, axis=-1)


class InceptionE(layers.Layer):
    def __init__(self):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2D(320, (1, 1))

        self.branch3x3_1 = BasicConv2D(384, (1, 1))
        self.branch3x3_2a = Sequential([
            layers.ZeroPadding2D((0, 1)),
            BasicConv2D(384, (1, 3))
        ])
        self.branch3x3_2b = Sequential([
            layers.ZeroPadding2D((1, 0)),
            BasicConv2D(384, (3, 1))
        ])
        self.branch3x3stack_1 = BasicConv2D(448, (1, 1))
        self.branch3x3stack_2 = BasicConv2D(384, (3, 3), padding='same')
        self.branch3x3stack_3a = Sequential([
            layers.ZeroPadding2D((0, 1)),
            BasicConv2D(384, (1, 3))
        ])
        self.branch3x3stack_3b = Sequential([
            layers.ZeroPadding2D((1, 0)),
            BasicConv2D(384, (3, 1))
        ])

        self.branchpool = Sequential([
            layers.AveragePooling2D((3, 3), strides=1, padding='same'),
            BasicConv2D(192, (1, 1))
        ])

    def call(self, x, training=False):
        branch1x1 = self.branch1x1(x, training=training)
        branch3x3 = self.branch3x3_1(x, training=training)
        branch3x3 = [
            self.branch3x3_2a(branch3x3, training=training),
            self.branch3x3_2b(branch3x3, training=training)
        ]
        branch3x3 = tf.concat(branch3x3, axis=-1)

        branch3x3stack = self.branch3x3stack_1(x, training=training)
        branch3x3stack = self.branch3x3stack_2(
            branch3x3stack, training=training)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack, training=training),
            self.branch3x3stack_3b(branch3x3stack, training=training)
        ]
        branch3x3stack = tf.concat(branch3x3stack, axis=-1)

        branchpool = self.branchpool(x, training=training)

        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]
        return tf.concat(outputs, axis=-1)


class InceptionV3(Model):
    def __init__(self, num_classes, input_shape=(32, 32, 3)):
        super(InceptionV3, self).__init__()

        self.conv1 = Sequential([
            layers.Input(input_shape),
            BasicConv2D(32, (3, 3), padding='same'),
            BasicConv2D(32, (3, 3), padding='same'),
            BasicConv2D(64, (3, 3), padding='same'),
            BasicConv2D(80, (1, 1)),
            BasicConv2D(192, (3, 3))
        ])
        self.conv2 = Sequential([
            InceptionA(32),
            InceptionA(64),
            InceptionA(64)
        ])
        self.conv3 = InceptionB()
        self.conv4 = Sequential([
            InceptionC(channels_7x7=128),
            InceptionC(channels_7x7=160),
            InceptionC(channels_7x7=160),
            InceptionC(channels_7x7=192)
        ])
        self.conv5 = InceptionD()
        self.conv6 = Sequential([
            InceptionE(),
            InceptionE()
        ])
        self.avgpool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.5)
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)
        x = self.avgpool(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)
        return x


def inceptionv3(num_classes):
    return InceptionV3(num_classes)
