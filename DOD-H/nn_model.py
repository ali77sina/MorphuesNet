import tensorflow as tf
import numpy as np

# Function that creates the model in the paper. This is the baseline used for all the metrics.
def separable_resnet(input_shape, num_classes, bias = False, y_train = [], reg_drop = False, lstm = False, PSD = False):
    # Input tensor shape
    inputs = tf.keras.layers.Input(shape=input_shape)
    kernel_sizes = [25,25,25]
    # Initial convolution layer
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 15), strides=(1, 2), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Max pooling layer
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)

    # Residual blocks
    for i in range(3): 
        # Separable convolution layer 1
        residual = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)    #orig value = 32
        residual = tf.keras.layers.BatchNormalization()(residual)
        residual = tf.keras.layers.Activation('relu')(residual)
        
        residual = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, kernel_sizes[i]), strides=(1, 1), padding='same', use_bias=False)(residual)
        residual = tf.keras.layers.BatchNormalization()(residual)
        residual = tf.keras.layers.Activation('relu')(residual)
        
        residual = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(residual)
        residual = tf.keras.layers.BatchNormalization()(residual)
        
        # Shortcut connection
        if i == 0:
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
        x = tf.keras.layers.Add()([x, residual])
        
        x = tf.keras.layers.Activation('relu')(x)
        if reg_drop:
            x = tf.keras.layers.Dropout(0.1)(x

    # Global average pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    if lstm:
        x = tf.keras.layers.Reshape((64,1))(x)
        x =  tf.keras.layers.LSTM(32)(x)
    if PSD:
        inputs2 = tf.keras.layers.Input(129)
        x2 = tf.keras.layers.Dense(32, 'relu')(inputs2)
        x = tf.keras.layers.concatenate([x2,x])
    x = tf.keras.layers.Dense(32, 'relu')(x)
    # Fully connected layer
    if bias:
        x = tf.keras.layers.Dense(units=num_classes, activation='softmax', bias_initializer = tf.keras.initializers.Constant(init_bias(y_train)))(x)
    else:
        x = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)

    # Define the model
    if PSD:
        model = tf.keras.Model(inputs=[inputs,inputs2], outputs=x)
    else:
        model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

# Sequence learner model, based on an LSTM layer.
def seq_model(input_shape):
    inp = tf.keras.layers.Input((input_shape,1))
    x = tf.keras.layers.LSTM(32)(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32,'relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(5, 'softmax')(x)
    model = tf.keras.models.Model(inputs = inp, outputs = out)
    return model
