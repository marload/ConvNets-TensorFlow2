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
            layers.Conv2D(squeeze_channel, int(out_channels / 2), 1),
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
    def __init__(self, num_classes):
        super(SqueezeNet, self).__init__()
        self.stem = Sequential([
            layers.Conv2D(96, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2), strides=2)
        ])

        self.fire2 = Fire(128, 16)
        self.fire3 = Fire(128, 16)
        self.fire4 = Fire(256, 32)
        self.fire5 = Fire(256, 32)
        self.fire6 = Fire(384, 48)
        self.fire7 = Fire(384, 48)
        self.fire8 = Fire(512, 64)
        self.fire9 = Fire(512, 64)

        self.conv10 = layers.Conv2D(num_classes, 1)
        self.gap = layers.GlobalAveragePooling2D()
        
