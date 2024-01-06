'''************************************************************************** 
ghostnet building blocks

ghostnet implementation
Author: fvilmos, https://github.com/fvilmos
***************************************************************************'''

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

def squeeze_excitation(input, output, se_ratio):
    """
    self attention function on the on chennels

    Args:
        input (tensor): tensor to be processes
        output (int): desired output channels
        ratio (float): scaling ratio

    Returns:
        tensor: SE value
    """
    new_filters = int(output * se_ratio)
    
    # squeeze
    y = keras.layers.GlobalAveragePooling2D()(input)
    y = keras.layers.Reshape((1,1,int(y.shape[1])))(y)
    y = keras.layers.Conv2D(filters=new_filters, kernel_size=(1,1), strides=(1,1), padding="SAME", activation='relu')(y)

    #excitation
    y = keras.layers.Conv2D(filters=output, kernel_size=(1,1), strides=(1,1), padding="SAME", activation='hard_sigmoid')(y)
    y = tf.multiply(y, tf.convert_to_tensor(tf.cast(se_ratio, dtype=tf.float32)))
    
    return y


def ghost_module(input, output,kernel_size=(1,1), dept_wise_kernel=(3,3), ratio=2, strides=(1,1), padding="SAME", activation='relu'):
    """
    Implements the ghost module, that generates ghost features, more feature maps from cheep operations.

    Args:
        input (tensor): input
        output (int): desired output channel size
        kernel_size (tuple, optional): Convalution kernel size. Defaults to (1,1).
        dept_wise_kernel (tuple, optional): Depth wise convolution kernel size. Defaults to (3,3).
        ratio (int, optional): Expansion ration. Defaults to 2.
        strides (tuple, optional): operation strides. Defaults to (1,1).
        padding (str, optional): Padding strategy. Defaults to "SAME".
        activation (str, optional): Activation function. Defaults to 'relu'.

    Returns:
        tensor: layers with desired size
    """
    out_channels = np.int(np.ceil(output/ratio))

    y0 = keras.layers.Conv2D(filters=out_channels,kernel_size=kernel_size,strides=strides, \
                            padding=padding)(input)
    y0 = keras.layers.BatchNormalization()(y0)

    if activation is not None:
        y0 = keras.layers.Activation(activation=activation)(y0)

    # handle cases where expansion is needed
    if ratio > 1:
        # cheap operation
        y1 = keras.layers.DepthwiseConv2D(kernel_size=dept_wise_kernel,strides=strides,
                                          padding=padding,
                                          depth_multiplier=ratio-1)(y0)
        
        # format channel size to the required output
        y1 = keras.layers.BatchNormalization()(y1)

        if activation is not None:
            y1 = keras.layers.Activation(activation=activation)(y1)

        out = keras.layers.concatenate([y0,y1], axis=-1)

        # format channel size to the required output
        out = out[...,:int(output)]
    else:
        out = y0

    return out

def ghost_bottleneck(input,expansion, output, use_se=False, strides=(1,1), kernel_size=(3,3), activation='relu', se_alpha=None, ratio=2):
    """
    Implements the channel expansion and reduction to the schortcut size, trough two ghost modules. 

    Args:
        input (tensor): input tensors
        expansion (int): channel expansion, used to up/downsize the ghost bottleneck
        output (int): channel output size
        use_se (bool, optional): _description_. Defaults to False.
        strides (tuple, optional): desired strides. Defaults to (1,1).
        kernel_size (tuple, optional): Kernel konvolution in gost modules. Defaults to (3,3).
        activation (str, optional): Used activation functions. Defaults to 'relu'.
        se_alpha (_type_, optional): Squeeze and excitation modul scaling ratio. Defaults to None.

    Returns:
        tensor: layes that implements the ghost bottleneck
    """

    # first ghost module
    g1 = ghost_module(input=input,output=expansion,
                      kernel_size=kernel_size,dept_wise_kernel=kernel_size,
                      ratio=ratio, strides=(1,1), activation='relu')
    # stride 2 case
    if strides[0] > 1:
        g1 = keras.layers.DepthwiseConv2D(kernel_size=kernel_size,strides=strides,
                                          padding="SAME",
                                          depth_multiplier=1)(g1)
        g1 = keras.layers.BatchNormalization()(g1)

    # use sqeeze - excitation function
    if (se_alpha is not None) and (se_alpha > 0):
        g1 = squeeze_excitation(input=g1,output=expansion, se_ratio=se_alpha)

    # secound ghost module
    g2 = ghost_module(input=g1,output=output,
                       kernel_size=kernel_size,dept_wise_kernel=kernel_size,
                       ratio=ratio, strides=(1,1), activation=None)

    g2 = keras.layers.BatchNormalization()(g2)
    
    # prepare input
    y = input

    # check channels is equal
    #if input.shape[-1] != output:
        #print ("ping")
    y = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='SAME', depth_multiplier=1)(input)
    y = keras.layers.BatchNormalization()(y)
    
    y = keras.layers.Conv2D(filters=output,kernel_size=(1,1), strides=(1,1), padding = 'SAME')(y)
    y = keras.layers.BatchNormalization()(y)

    # return the added layers
    return keras.layers.add([y, g2])
