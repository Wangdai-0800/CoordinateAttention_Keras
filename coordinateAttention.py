# A keras implementation of Coordinate Attention follows https://github.com/Andrew-Qibin/CoordAttention
# Created by Wang Dai on Sept. 24, 2021
import tensorflow as tf
import tensorflow.keras
import keras.backend as K
from tensorflow.keras.layers import AveragePooling2D, Permute, Lambda, multiply
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, ReLU

def coordinateAttentionLayer(x, inputChannel, outputChannel, reductionRatio=32):
    def h_swish(x):
        return ReLU(6., name="ReLU6_1")(x+3)/6

    #Hope input x has a shape of NHWC
    identity = x
    [n, h, w, c] = x.shape
    x_h = AveragePooling2D(pool_size=(h, 1), strides=1,
                           padding='valid',
                           data_format="channels_last")(x)
    x_w = AveragePooling2D(pool_size=(1, w), strides=1,
                           padding='valid',
                           data_format="channels_last")(x)
    x_w = Permute((2, 1, 3))(x_w)
    y = K.concatenate((x_h, x_w), axis=2)
    reductionChannel = max(8, inputChannel//reductionRatio)
    y = Conv2D(filters=reductionChannel, kernel_size=1,
               strides=1, padding="valid")(y)
    y = BatchNormalization()(y)
    y = h_swish(y)
    x_h, x_w = Lambda(tf.split, arguments={"axis":2, "num_or_size_splits":[w, h]})(y)
    x_w = Permute((2, 1, 3))(x_w)

    a_h = Conv2D(filters=outputChannel, kernel_size=1,
                 strides=1, padding="valid", activation="sigmoid")(x_h)
    a_w = Conv2D(filters=outputChannel, kernel_size=1,
                 strides=1, padding="valid", activation="sigmoid")(x_w)
    a_h = tf.tile(a_h, [1, h, 1, 1])
    a_w = tf.tile(a_w, [1, 1, w, 1])
    out = multiply([identity, a_w, a_h])
    return out



