import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

class Fire(layers.Layer):
    def __init__(self, out_channels, squeeze_channel):
        super(Fire, self).__init__()
        self.squeeze = Sequential([
            layers.Conv2D(squeeze_channel, (1, 1)),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.expand_1x1 = Sequential([
            layers.Conv2D(int(out_channels / 2), (1, 1)),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.expand_3x3 = Sequential([
            layers.Conv2D(int(out_channels / 2), (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
    
    def call(self, x, training=False):
        x = self.squeeze(x, training=training)
        x = tf.concat([
            self.expand_1x1(x, training=training),
            self.expand_3x3(x, training=training)
        ], -1)

        return x
    

class SqueezeNet(Model):
    def __init__(self, num_classes, input_shape=(32, 32, 3)):
        super(SqueezeNet, self).__init__()
        self.stem = Sequential([
            layers.Input(input_shape),
            layers.Conv2D(96, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2), strides=2)
        ])
        self.fire = Sequential([
            Fire(128, 16),
            Fire(128, 16),
            Fire(256, 32),
            Fire(256, 32),
            Fire(384, 48),
            Fire(384, 48),
            Fire(512, 64),
            Fire(512, 64)
        ])
        self.conv = layers.Conv2D(num_classes, 1)
        self.ap = layers.AveragePooling2D((7, 7), strides=1)
        self.mp = layers.MaxPooling2D()
        self.flat = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.stem(inputs, training=training)
        x = self.fire(x, training=training)
        x = self.conv(x, training=training)
        x = self.ap(x)
        x = self.mp(x)
        x = self.flat(x)
        x = self.fc(x)
        return x

def squeezenet(num_classes):
    return SqueezeNet(num_classes)
