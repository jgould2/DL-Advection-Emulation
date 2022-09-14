import keras.layers
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
def model1(pretrained_weights=None, input_size=(256, 256, 6)):
    inputs = Input(input_size)
    input = Lambda(lambda b: tf.slice(b, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    conv1 = Conv2D(1536, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(inputs)
    conv2 = Conv2D(1536, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv1)
    conv3 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv2)
    conv4 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv3)
    conv5 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv4)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)
    conv7 = Conv2D(1, 1)(conv6)
    out = Add()([input, conv7])
    model = Model(inputs=inputs, outputs=out, name='FCN_narrow_1')

    model.compile(optimizer=Adam(lr=1e-4), loss='mae', metrics=['mae'])

    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#########################################################################################################################################

def model2(pretrained_weights=None, input_size=(256, 256, 6)):
    inputs = Input(input_size)
    input = Lambda(lambda b: tf.slice(b, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    velocities = Lambda(lambda b: tf.slice(b, [0, 0, 0, 1], [-1, -1, -1, -1]))(inputs)
    v_times_in = Lambda(lambda b: concatenate([b[0] * tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(5)], axis=-1))([input, velocities])
    conv1 = Conv2D(1536, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_times_in)
    conv2 = Conv2D(1536, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv1)
    conv3 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv2)
    conv4 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv3)
    conv5 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv4)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)
    conv7 = Conv2D(1, 1)(conv6)
    out = Add()([input, conv7])
    model = Model(inputs=inputs, outputs=out, name='FCN_narrow_2')

    model.compile(optimizer=Adam(lr=1e-4), loss='mae', metrics=['mae'])

    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#########################################################################################################################################

def model3(pretrained_weights=None, input_size=(256, 256, 6)):
    inputs = Input(input_size)
    input = Lambda(lambda b: tf.slice(b, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    velocities = Lambda(lambda b: tf.slice(b, [0, 0, 0, 1], [-1, -1, -1, -1]))(inputs)
    v_times_in = Lambda(lambda b: concatenate([b[0] * tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(5)], axis=-1))([input, velocities])
    conv1 = Conv2D(1536, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_times_in)
    conv2 = Conv2D(1536, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv1)
    v_x_f_1 = Lambda(lambda b: concatenate([tf.slice(b[1], [0, 0, 0, int(i * 1536 / 5)], [-1, -1, -1, int(1536 / 5)]) * tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(5)], axis=-1))([conv2, velocities])
    conv3 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_x_f_1)
    conv4 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv3)
    v_x_f_2 = Lambda(lambda b: concatenate([tf.slice(b[1], [0, 0, 0, int(i * 1024 / 5)], [-1, -1, -1, int(1024 / 5)]) * tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(5)], axis=-1))([conv4, velocities])
    conv5 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_x_f_2)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)
    v_x_f_3 = Lambda(lambda b: concatenate([tf.slice(b[1], [0, 0, 0, int(i * 512 / 5)], [-1, -1, -1, int(512 / 5)]) * tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(5)], axis=-1))([conv6, velocities])
    concat_1 = concatenate([v_x_f_1, v_x_f_2, v_x_f_3], axis=-1)
    conv7 = Conv2D(1, 1)(concat_1)
    out = Add()([input, conv7])
    model = Model(inputs=inputs, outputs=out, name='FCN_narrow_3')

    model.compile(optimizer=Adam(lr=1e-4), loss='mae', metrics=['mae'])
    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#########################################################################################################################################

def model4(pretrained_weights=None, input_size=(64, 64, 3)):
    inputs = Input(input_size)
    input = Lambda(lambda b: tf.slice(b, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    conv1 = Conv2D(1536, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(inputs)
    conv2 = Conv2D(1536, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv1)
    conv3 = Conv2D(384, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv2)
    conv4 = Conv2D(384, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv3)
    conv5 = Conv2D(96, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv4)
    conv6 = Conv2D(96, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)
    conv7 = Conv2D(1, 1)(conv6)
    out = Add()([input, conv7])
    model = Model(inputs=inputs, outputs=out, name='FCN_narrow_4')

    model.compile(optimizer=Adam(lr=1e-4), loss='mae', metrics=['mae'])

    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#########################################################################################################################################

def model5(pretrained_weights=None, input_size=(64, 64, 3)):
    inputs = Input(input_size)
    input = Lambda(lambda b: tf.slice(b, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    velocities = Lambda(lambda b: tf.slice(b, [0, 0, 0, 1], [-1, -1, -1, -1]))(inputs)
    v_times_in = Lambda(lambda b: concatenate([b[0] * tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)], axis=-1))([input, velocities])
    conv1 = Conv2D(1536, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_times_in)
    conv2 = Conv2D(1536, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv1)
    conv3 = Conv2D(384, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv2)
    conv4 = Conv2D(384, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv3)
    conv5 = Conv2D(96, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv4)
    conv6 = Conv2D(96, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)
    conv7 = Conv2D(1, 1)(conv6)
    out = Add()([input, conv7])
    model = Model(inputs=inputs, outputs=out, name='FCN_narrow_5')

    model.compile(optimizer=Adam(lr=1e-4), loss='mae', metrics=['mae'])

    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#########################################################################################################################################

def model6(pretrained_weights=None, input_size=(64, 64, 3)):
    inputs = Input(input_size)
    input = Lambda(lambda b: tf.slice(b, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    velocities = Lambda(lambda b: tf.slice(b, [0, 0, 0, 1], [-1, -1, -1, -1]))(inputs)
    v_times_in = Lambda(lambda b: concatenate([b[0] * tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)], axis=-1))([input, velocities])
    conv1 = Conv2D(1536, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_times_in)
    conv2 = Conv2D(1536, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv1)
    v_x_f_1 = Lambda(lambda b: concatenate([tf.slice(b[0], [0, 0, 0, int(i * 1536 / 2)], [-1, -1, -1, int(1536 / 2)]) * tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)], axis=-1))([conv2, velocities])
    conv3 = Conv2D(384, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_x_f_1)
    conv4 = Conv2D(384, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv3)
    v_x_f_2 = Lambda(lambda b: concatenate([tf.slice(b[0], [0, 0, 0, int(i * 384 / 2)], [-1, -1, -1, int(384 / 2)]) * tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)], axis=-1))([conv4, velocities])
    conv5 = Conv2D(96, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_x_f_2)
    conv6 = Conv2D(96, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)
    v_x_f_3 = Lambda(lambda b: concatenate([tf.slice(b[0], [0, 0, 0, int(i * 96 / 2)], [-1, -1, -1, int(96 / 2)]) * tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)], axis=-1))([conv6, velocities])
    concat_1 = concatenate([v_x_f_1, v_x_f_2, v_x_f_3], axis=-1)
    conv7 = Conv2D(1, 1)(concat_1)
    out = Add()([input, conv7])
    model = Model(inputs=inputs, outputs=out, name='FCN_narrow_6')

    model.compile(optimizer=Adam(lr=1e-4), loss='mae', metrics=['mae'])
    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
#########################################################################################################################################
