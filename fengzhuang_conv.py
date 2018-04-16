from functools import wraps

import tensorflow as tf
from keras import backend as k
from keras.layers import Conv2D, Add, UpSampling2D, Concatenate
from keras.regularizers import l2


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):

    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4),
                           'padding': 'same',
                           }
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **kwargs)


