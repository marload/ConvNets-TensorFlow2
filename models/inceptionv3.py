import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

class BasicConv2D(layers.Layer):
    def __init__(self,
                 kernels, 
                 kernel_size=(3, 3), 
                 strdies=1, 
                 padding='valid',
                 use_bias=False):
        super(BasicConv2D, self).__init__()
        self.conv = layers.Conv2D(kernels,
                                  kernel_size,
                                  strides=strdies, 
                                  padding=padding, 
                                  use_bias=use_bias)
        self.bn = layers.BatchNormalization()
    
    def call(self, x, training=training):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)


class InceptionA(layers.Layer):
    def __init__(self, pool_features):
        super(InceptionA, self).__init__()
        
        self.branch1x1 = BasicConv2D(64, (1, 1))
        
        self.branch5x5 = Sequential([
            BasicConv2D(48, (1, 1), padding='same'),
            BasicConv2D(64, (5, 5), padding='same')
        ])
        self.branch3x3 = Sequential([
            BasicConv2D(64, (1, 1), padding='same'),
            BasicConv2D(96, (3, 3), padding='same'),
            BasicConv2D(96, (3, 3), padding='same'),
        ])
        self.branchpool = Sequential([
            layers.AveragePool2D((3, 3), strides=1, padding='same'),
            BasicConv2D(pool_features, (3, 3), padding='same')
        ])
    
    def forward(self, x, training=False):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3 = self.branch3x3(x)
        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]
        
        return tf.concat(outputs, axis=1)
    
    
