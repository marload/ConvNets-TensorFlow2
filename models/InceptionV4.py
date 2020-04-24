import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

class BasicConv2D(layers.Layer):
    def __init__(self, kernels, kernel_size=(3, 3), strides=1, padding='valid'):
        super(BasicConv2D, self).__init__()
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
    

class Inception_Stem(layers.Layer):
    def __init__(self, input_shape=(32, 32, 3)):
        super(Inception_Stem, self).__init__()
        self.conv1 = Sequential([
            layers.Input(input_shape),
            BasicConv2D(32, (3, 3)),
            BasicConv2D(32, (3, 3), padding='same'),
            BasicConv2D(64, (3, 3), padding='same')
        ])
        self.branch3x3_conv = BasicConv2D(96, (3, 3), padding='same')
        self.branch3x3_pool = layers.MaxPooling2D((3, 3), strides=1, padding='same')
        self.branch7x7a = Sequential([
            BasicConv2D(64, (1, 1)),
            layers.ZeroPadding2D((3, 0)),
            BasicConv2D(64, (7, 1)),
            layers.ZeroPadding2D((0, 3)),
            BasicConv2D(64, (1, 7)),
            BasicConv2D(96, (3, 3), padding='same')
        ])
        self.branch7x7b = Sequential([
            BasicConv2D(64, (1, 1)),
            BasicConv2D(96, (3, 3), padding='same')
        ])
        self.branchpoola = layers.MaxPooling2D((3, 3), strides=1, padding='same')
        self.branchpoolb = BasicConv2D(192, (3, 3), padding='same')
    
    def call(self, x, training=False):
        x = self.conv1(x, training=training)

        x = [
            self.branch3x3_conv(x, training=training),
            self.branch3x3_pool(x, training=training)
        ]
        x = tf.concat(x, axis=-1)

        x = [
            self.branch7x7a(x, training=training),
            self.branch7x7b(x, training=training)
        ]
        x = tf.concat(x, axis=-1)

        return x

class InceptionA(layers.Layer):
    def __init__(self):
        super(InceptionA, self).__init__()

        self.branch3x3stack = Sequential([
            BasicConv2D(64, (1, 1)),
            BasicConv2D(96, (3, 3), padding='same'),
            BasicConv2D(96, (3, 3), padding='same')
        ])
        self.branch3x3 = Sequential([
            BasicConv2D(64, (1, 1)),
            BasicConv2D(96, (3, 3), padding='same')
        ])
        self.branch1x1 = BasicConv2D(96, (1, 1))
        self.branchpool = Sequential([
            layers.AveragePooling2D((3, 3), strides=1, padding='same'),
            BasicConv2D(96, (1, 1))
        ])

    def call(self, x, training=False):
        x = [
            self.branch3x3stack(x, training=training),
            self.branch3x3(x, training=training),
            self.branch1x1(x, training=training),
            self.branchpool(x, training=training)
        ]
        return tf.concat(x, axis=-1)

class ReductionA(layers.Layer):
    def __init__(self, k, l, m, n):
        super(ReductionA, self).__init__()

        self.branch3x3stack = Sequential([
            BasicConv2D(k, (1, 1)),
            BasicConv2D(l, (3, 3), padding='same'),
            BasicConv2D(m, (3, 3), strides=2)
        ])
        self.branch3x3 = BasicConv2D(n, (3, 3), strides=2)
        self.branchpool = layers.MaxPooling2D((3, 3), strides=2)
    
    def call(self, x, training=False):
        x = [
            self.branch3x3stack(x, training=training),
            self.branch3x3(x, training=training),
            self.branchpool(x, training=training)
        ]
        return tf.concat(x, axis=-1)

class InceptionB(layers.Layer):
    def __init__(self):
        super(InceptionB, self).__init__()

        self.branch7x7stack = Sequential([
            BasicConv2D(192, (1, 1)),
            layers.ZeroPadding2D((0, 3)),
            BasicConv2D(192, (1, 7)),
            layers.ZeroPadding2D((3, 0)),
            BasicConv2D(224, (7, 1)),
            layers.ZeroPadding2D((0, 3)),
            BasicConv2D(224, (1, 7)),
            layers.ZeroPadding2D((3, 0)),
            BasicConv2D(256, (7, 1))
        ])
        self.branch7x7 = Sequential([
            BasicConv2D(192, (1, 1)),
            layers.ZeroPadding2D((0, 3)),
            BasicConv2D(224, (1, 7)),
            layers.ZeroPadding2D((3, 0)),
            BasicConv2D(256, (7, 1))
        ])
        self.branch1x1 = BasicConv2D(384, (1, 1)) 

        self.branchpool = Sequential([
            layers.AveragePooling2D((3, 3), strides=1, padding='same'),
            BasicConv2D(128, (1, 1))
        ])
    
    def call(self, x, training=False):
        x = [
            self.branch1x1(x, training=training),
            self.branch7x7(x, training=training),
            self.branch7x7stack(x, training=training),
            self.branchpool(x, training=training)
        ]

        return tf.concat(x, axis=-1)


class ReductionB(layers.Layer):
    def __init__(self):
        super(ReductionB, self).__init__()

        self.branch7x7 = Sequential([
            BasicConv2D(256, (1, 1)),
            layers.ZeroPadding2D((0, 3)),
            BasicConv2D(256, (1, 7)),
            layers.ZeroPadding2D((3, 0)),
            BasicConv2D(320, (7, 1)),
            layers.ZeroPadding2D((1, 1)),
            BasicConv2D(320, (3, 3), strides=2)
        ])
        self.branch3x3 = Sequential([
            BasicConv2D(192, (1, 1)),
            layers.ZeroPadding2D((1, 1)),
            BasicConv2D(192, (3, 3), strides=2)
        ])
        self.branchpool = Sequential([
            layers.ZeroPadding2D((1, 1)),
            layers.MaxPooling2D((3, 3), strides=2)
        ])
    
    def call(self, x, training=False):
        x = [
            self.branch3x3(x, training=training),
            self.branch7x7(x, training=training),
            self.branchpool(x, training=training)
        ]

        return tf.concat(x, axis=-1)


class InceptionC(layers.Layer):
    def __init__(self):
        super(InceptionC, self).__init__()

        self.branch3x3stack = Sequential([
            BasicConv2D(384, (1, 1)),
            layers.ZeroPadding2D((0, 1)),
            BasicConv2D(448, (1, 3)),
            layers.ZeroPadding2D((1, 0)),
            BasicConv2D(512, (3, 1))
        ])
        self.branch3x3stacka = Sequential([
            layers.ZeroPadding2D((0, 1)),
            BasicConv2D(256, (1, 3))
        ])
        self.branch3x3stackb = Sequential([
            layers.ZeroPadding2D((1, 0)),
            BasicConv2D(256, (3, 1))
        ])
    
        self.branch3x3 = BasicConv2D(384, (1, 1))
        self.branch3x3a = Sequential([
            layers.ZeroPadding2D((1, 0)),
            BasicConv2D(256, (3, 1))
        ])
        self.branch3x3b = Sequential([
            layers.ZeroPadding2D((0, 1)),
            BasicConv2D(256, (1, 3))
        ])
        self.branch1x1 = BasicConv2D(256, (1, 1))
        self.branchpool = Sequential([
            layers.AveragePooling2D((3, 3), strides=1, padding='same'),
            BasicConv2D(256, (1, 1))
        ])

    def forward(self, x, training=False):
        branch3x3stack_output = self.branch3x3stack(x, training=training)
        branch3x3stack_output = [
            self.branch3x3stacka(branch3x3stack_output, training=training),
            self.branch3x3stackb(branch3x3stack_output, training=training)
        ]
        branch3x3stack_output = tf.concat(branch3x3stack_output, axis=-1)

        branch3x3_output = self.branch3x3(x, training=training)
        branch3x3_output = [
            self.branch3x3a(branch3x3_output, training=training),
            self.branch3x3b(branch3x3_output, training=training)
        ]
        branch3x3_output = tf.concat(branch3x3_output, axis=-1)
        branch1x1_output = self.branch1x1(x, training=training)
        branchpool = self.branchpool(x, training=training)

        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]

        return tf.concat(output, axis=-1)


class InceptionV4(Model):
    def __init__(self, 
                 num_classes, 
                 A, B, C, 
                 k=192, l=224, 
                 m=256, n=384, 
                 input_shape=(32, 32, 3)):
        super(InceptionV4, self).__init__()

        self.stem = Inception_Stem(input_shape)
        self.inception_a = self._generate_inception_module(384, A, InceptionA)
        self.reduction_a = ReductionA(k, l, m, n)
        self.inception_b = self._generate_inception_module(1024, B, InceptionB)
        self.reduction_b = ReductionB()
        self.inception_c = self._generate_inception_module(1536, C, InceptionC)
        self.ap = layers.AveragePooling2D((7, 7)) # TODO strides check

        self.dropout = layers.Dropout(0.2)
        self.flat = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.stem(inputs, training=False)
        x = self.inception_a(x, training=False)
        x = self.reduction_a(x, training=False)
        x = self.inception_b(x, training=False)
        x = self.reduction_b(x, training=False)
        x = self.inception_c(x, training=False)
        x = self.ap(x)
        x = self.dropout(x, training=False)
        x = self.flat(x)
        x = self.fc(x)
        return x
    
    @staticmethod
    def _generate_inception_module(out_channels, block_num, block):
        nets = Sequential()
        for l in range(block_num):
            nets.add(block())
        return nets
    
class InceptionResNetA(layers.Layer):
    def __init__(self):
        super(InceptionResNetA, self).__init__()

        self.branch3x3stack = Sequential([
            BasicConv2D(32, (1, 1)),
            BasicConv2D(48, (3, 3), padding='same'),
            BasicConv2D(64, (3, 3), padding='same')
        ])
        self.branch3x3 = Sequential([
            BasicConv2D(32, (1, 1)),
            BasicConv2D(32, (3, 3), padding='same')
        ])
        self.branch1x1 = BasicConv2D(32, (1, 1))

        self.reduction1x1 = layers.Conv2D(384, (1, 1))
        self.shortcut = layers.Conv2D(384, (1, 1))
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        residual = [
            self.branch1x1(x, training=training),
            self.branch3x3(x, training=training),
            self.branch3x3stack(x, training=training),
        ]

        residual = tf.concat(residual, axis=-1)
        residual = self.reduction1x1(residual, training=training)
        shortcut = self.shortcut(x, training=training)

        output = self.bn(shortcut + residual)
        output = self.relu(output)
        
        return output


class InceptionResNetB(layers.Layer):
    def __init__(self):
        super(InceptionResNetB, self).__init__()
        self.branch7x7 = Sequential([
            BasicConv2D(128, (1, 1)),
            layers.ZeroPadding2D((0, 3)),
            BasicConv2D(160, (1, 7)),
            layers.ZeroPadding2D((3, 0)),
            BasicConv2D(192, (7, 1)),
        ])
        self.branch1x1 = BasicConv2D(192, (1, 1))
        self.reduction1x1 = layers.Conv2D(1154, (1, 1))
        self.shortcut = layers.Conv2D(1154, (1, 1))

        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
    
    def call(self, x, training=False):
        residual = [
            self.branch1x1(x, training=training),
            self.branch7x7(x, training=training)
        ]

        residual = tf.concat(residual, axis=-1)
        residual = self.reduction1x1(residual) * 0.1

        shortcut = self.shortcut(x, training=training)

        output = self.bn(residual + shortcut, training=training)
        output = self.relu(output)

        return output


class InceptionResNetC(layers.Layer):
    def __init__(self):
        super(InceptionResNetC, self).__init__()

        self.branch3x3 = Sequential([
            BasicConv2D(192, (1, 1)),
            layers.ZeroPadding2D((0, 1)),
            BasicConv2D(224, (1, 3)),
            layers.ZeroPadding2D((1, 0)),
            BasicConv2D(256, (3, 1))
        ])
        self.branch1x1 = BasicConv2D(192, (1, 1))
        self.reduction1x1 = layers.Conv2D(2048, (1, 1))
        self.shortcut = layers.Conv2D(2048, (1, 1))
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        residual = [
            self.branch1x1(x, training=training),
            self.branch3x3(x, training=training)
        ]

        residual = tf.concat(residual, axis=-1)
        residual = self.reduction1x1(residual, training=training) * 0.1

        shortcut = self.shortcut(x, training=training)

        output = self.bn(shortcut + residual)
        output = self.relu(output)

        return output


class InceptionResNetReductionA(layers.Layer):
    def __init__(self, k, l, m, n):
        super(InceptionResNetReductionA, self).__init__()

        self.branch3x3stack = Sequential([
            BasicConv2D(k, (1, 1)),
            BasicConv2D(l, (3, 3), padding='same'),
            BasicConv2D(m, (3, 3), strides=2)
        ])
        self.branch3x3 = BasicConv2D(n, (3, 3), strides=2)
        self.branchpool = layers.MaxPooling2D((3, 3), strides=2)
    
    def call(self, x, training=False):
        x = [
            self.branch3x3stack(x, training=training),
            self.branch3x3(x, training=training),
            self.branchpool(x, training=training)
        ]

        return tf.concat(x, axis=-1)

class InceptionResNetReductionB(layers.Layer):
    def __init__(self):
        super(InceptionResNetReductionB, self).__init__()

        self.branchpool = layers.MaxPooling2D((3, 3), strides=2)
        self.branch3x3a = Sequential([
            BasicConv2D(256, (1, 1)),
            BasicConv2D(384, (3, 3), strides=2)
        ])
        self.branch3x3b = Sequential([
            BasicConv2D(256, (1, 1)),
            BasicConv2D(288, (3, 3), strides=2)
        ])
        self.branch3x3stack = Sequential([
            BasicConv2D(256, (1, 1)),
            BasicConv2D(288, (3, 3), padding='same'),
            BasicConv2D(320, (3, 3), strides=2)
        ])

    def call(self, x, training=False):
        x = [
            self.branch3x3a(x, training=training),
            self.branch3x3b(x, training=training),
            self.branch3x3stack(x, training=training),
            self.branchpool(x, training=training)
        ]

        return tf.concat(x, axis=-1)


class InceptionResNetV2(Model):
    def __init__(self,
                 num_classes, 
                 A, B, C , 
                 k=256, l=256, m=384, n=384, 
                 input_shape=(32, 32, 3)):
        super(InceptionResNetV2, self).__init__()

        self.stem = Inception_Stem(input_shape)
        self.inception_resnet_a = self._generate_inception_module(384, A, InceptionResNetA)
        self.reduction_a = InceptionResNetReductionA(k, l, m, n)
        self.inception_resnet_b = self._generate_inception_module(1154, B, InceptionResNetB)
        self.reduction_b = InceptionResNetReductionB()
        self.inception_resnet_c = self._generate_inception_module(2048, C, InceptionResNetC)

        self.ap = layers.AveragePooling2D((1, 1))
        self.dropout = layers.Dropout(0.2)
        self.flat = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
    
    @staticmethod
    def _generate_inception_module(output_channels, block_num, block):
        nets = Sequential()
        for l in range(block_num):
            nets.add(block())
        return nets

    def call(self, inputs, training=False):
        x = self.stem(inputs, training=training)
        x = self.inception_resnet_a(x, training=training)
        x = self.reduction_a(x, training=training)
        x = self.inception_resnet_b(x, training=training)
        x = self.reduction_b(x, training=training)
        x = self.inception_resnet_c(x, training=training)
        x = self.ap(x)
        x = self.dropout(x, training=training)
        x = self.flat(x)
        x = self.fc(x)

        return x

def inceptionv4(num_classes):
    return InceptionV4(num_classes, 4, 7, 3)

def inception_resnet_v2(num_classes):
    return InceptionResNetV2(num_classes, 5, 10, 5)