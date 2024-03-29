from keras.models import *
from keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
def model1(pretrained_weights=None, input_shape=(232, 396, 5)):
    inputs = Input(input_shape)
    input = Lambda(lambda b: tf.slice(b, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    conv1 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    padd3 = Lambda(lambda b: tf.pad(b, paddings = tf.constant([[0, 0], [1, 1], [1, 0], [0, 0]])))(pool2)
    conv3 = Conv2D(256, 3, activation='tanh', padding='valid', kernel_initializer='he_normal')(padd3)
    conv3 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    padd4 = Lambda(lambda b: tf.pad(b, paddings = tf.constant([[0, 0], [1, 0], [0, 1], [0, 0]])))(pool3)
    conv4 = Conv2D(512, 3, activation='tanh', padding='valid', kernel_initializer='he_normal')(padd4)
    conv4 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([conv4, up6], axis=3)
    padd6 = Lambda(lambda b: tf.pad(b, paddings = tf.constant([[0, 0], [1, 0], [0, 1], [0, 0]])))(merge6)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(padd6)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    padd7 = Lambda(lambda b: tf.pad(b, paddings=tf.constant([[0, 0], [0, 0], [1, 0], [0, 0]])))(merge7)
    conv7 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(padd7)
    conv7 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1)(conv9)
    out = Add()([conv10, input])
    model = Model(inputs=inputs, outputs=out, name='Unet_1')

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='mae', metrics=['mae'])

    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

######################################################################################################################

def model2(pretrained_weights=None, input_shape=(232, 396, 5)):
    inputs = Input(input_size)
    input = Lambda(lambda b: tf.slice(b, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    velocities = Lambda(lambda b: tf.slice(b, [0, 0, 0, 1], [-1, -1, -1, -1]))(inputs)
    v_times_in = Lambda(lambda b: concatenate([b[0]*tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(5)], axis=-1))([input, velocities])
    conv1 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_times_in)
    conv1 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    padd3 = Lambda(lambda b: tf.pad(b, paddings = tf.constant([[0, 0], [1, 1], [1, 0], [0, 0]])))(pool2)
    conv3 = Conv2D(256, 3, activation='tanh', padding='valid', kernel_initializer='he_normal')(padd3)
    conv3 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    padd4 = Lambda(lambda b: tf.pad(b, paddings = tf.constant([[0, 0], [1, 0], [0, 1], [0, 0]])))(pool3)
    conv4 = Conv2D(512, 3, activation='tanh', padding='valid', kernel_initializer='he_normal')(padd4)
    conv4 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([conv4, up6], axis=3)
    padd6 = Lambda(lambda b: tf.pad(b, paddings = tf.constant([[0, 0], [1, 0], [0, 1], [0, 0]])))(merge6)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(padd6)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    padd7 = Lambda(lambda b: tf.pad(b, paddings=tf.constant([[0, 0], [0, 0], [1, 0], [0, 0]])))(merge7)
    conv7 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(padd7)
    conv7 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1)(conv9)
    out = Add()([conv10, input])
    model = Model(inputs=inputs, outputs=out, name='Unet_1')

    model.compile(optimizer=Adam(lr=1e-4), loss='mae', metrics=['mae'])

    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#########################################################################################################################################

def model3(pretrained_weights=None, input_shape=(232, 396, 5)):
    inputs = Input(input_size)
    input = Lambda(lambda b: tf.slice(b, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    velocities = Lambda(lambda b: tf.slice(b, [0, 0, 0, 1], [-1, -1, -1, -1]))(inputs)
    v_times_in = Lambda(lambda b: concatenate([b[0]*tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(5)], axis=-1))([input, velocities])
    conv1 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_times_in)
    conv1 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    pool1_1 = AveragePooling2D(pool_size=(2, 2))(velocities)
    v_x_f_1 = Lambda(lambda b: concatenate([tf.slice(b[1], [0, 0, 0, int(i * 64 / 5)],
                                                     [-1, -1, -1, int(64 / 5)]) * tf.slice(b[1], [0, 0, 0, i],
                                                                                           [-1, -1, -1, 1]) for i in
                                            range(5)], axis=-1))([pool1, pool1_1])
    conv2 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_x_f_1)
    conv2 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    pool2_1 = AveragePooling2D(pool_size=(2, 2))(pool1_1)
    v_x_f_2 = Lambda(lambda b: concatenate([tf.slice(b[1], [0, 0, 0, int(i * 128 / 5)],
                                                     [-1, -1, -1, int(128 / 5)]) * tf.slice(b[1], [0, 0, 0, i],
                                                                                            [-1, -1, -1, 1]) for i in
                                            range(5)], axis=-1))([pool2, pool2_1])
    padd3 = Lambda(lambda b: tf.pad(b, paddings = tf.constant([[0, 0], [1, 1], [1, 0], [0, 0]])))(v_x_f_2)
    conv3 = Conv2D(256, 3, activation='tanh', padding='valid', kernel_initializer='he_normal')(padd3)
    conv3 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    pool3_1 = AveragePooling2D(pool_size=(2, 2))(pool2_1)
    v_x_f_3 = Lambda(lambda b: concatenate([tf.slice(b[1], [0, 0, 0, int(i * 256 / 5)],
                                                     [-1, -1, -1, int(256 / 5)]) * tf.slice(b[1], [0, 0, 0, i],
                                                                                            [-1, -1, -1, 1]) for i in
                                            range(5)], axis=-1))([pool3, pool3_1])
    padd4 = Lambda(lambda b: tf.pad(b, paddings = tf.constant([[0, 0], [1, 0], [0, 1], [0, 0]])))(v_x_f_3)
    conv4 = Conv2D(512, 3, activation='tanh', padding='valid', kernel_initializer='he_normal')(padd4)
    conv4 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)
    pool4_1 = AveragePooling2D(pool_size=(2, 2))(pool3_1)
    v_x_f_4 = Lambda(lambda b: concatenate([tf.slice(b[1], [0, 0, 0, int(i * 512 / 5)],
                                                     [-1, -1, -1, int(512 / 5)]) * tf.slice(b[1], [0, 0, 0, i],
                                                                                            [-1, -1, -1, 1]) for i in
                                            range(5)], axis=-1))([pool4, pool4_1])
    conv5 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_x_f_4)
    conv5 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([conv4, up6], axis=3)
    padd6 = Lambda(lambda b: tf.pad(b, paddings = tf.constant([[0, 0], [1, 0], [0, 1], [0, 0]])))(merge6)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(padd6)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    padd7 = Lambda(lambda b: tf.pad(b, paddings=tf.constant([[0, 0], [0, 0], [1, 0], [0, 0]])))(merge7)
    conv7 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(padd7)
    conv7 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1)(conv9)
    out = Add()([conv10, input])
    model = Model(inputs=inputs, outputs=out, name='Unet_1')

    model.compile(optimizer=Adam(lr=1e-4), loss='mae', metrics=['mae'])

    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#########################################################################################################################################

def model4(pretrained_weights=None, input_shape=(64, 64, 3)):
    inputs = Input(input_size)
    input = Lambda(lambda b: tf.slice(b, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    conv1 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(48, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1)(conv9)
    out = Add()([conv10, input])
    model = Model(inputs=inputs, outputs=out, name='Unet_4')

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='mae', metrics=['mae'])

    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#########################################################################################################################################

def model5(pretrained_weights=None, input_shape=(64, 64, 3)):
    inputs = Input(input_size)
    input = Lambda(lambda b: tf.slice(b, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    velocities = Lambda(lambda b: tf.slice(b, [0, 0, 0, 1], [-1, -1, -1, -1]))(inputs)
    v_times_in = Lambda(lambda b: concatenate([b[0]*tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)], axis=-1))([input, velocities])
    conv1 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_times_in)
    conv1 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(48, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(4, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1)(conv9)
    out = Add()([conv10, input])
    model = Model(inputs=inputs, outputs=out, name='Unet_5')

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss='mae', metrics=['mae'])

    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#########################################################################################################################################

def model6(pretrained_weights=None, input_shape=(64, 64, 3)):
    inputs = Input(input_shape)
    input = Lambda(lambda b: tf.slice(b, [0, 0, 0, 0], [-1, -1, -1, 1]))(inputs)
    velocities = Lambda(lambda b: tf.slice(b, [0, 0, 0, 1], [-1, -1, -1, -1]))(inputs)
    v_times_in = Lambda(lambda b: concatenate([b[0]*tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)], axis=-1))([input, velocities])
    conv1 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_times_in)
    conv1 = Conv2D(64, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    pool1_1 = AveragePooling2D(pool_size=(2, 2))(velocities)
    v_x_f_1 = Lambda(lambda b: concatenate([tf.slice(b[0], [0, 0, 0, int(i*64/2)], [-1, -1, -1, int(64/2)])*tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)], axis=-1))([pool1, pool1_1])
    conv2 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_x_f_1)
    conv2 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    pool2_1 = AveragePooling2D(pool_size=(2, 2))(pool1_1)
    v_x_f_2 = Lambda(lambda b: concatenate([tf.slice(b[0], [0, 0, 0, int(i*128/2)], [-1, -1, -1, int(128/2)])*tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)], axis=-1))([pool2, pool2_1])
    conv3 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_x_f_2)
    conv3 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    pool3_1 = AveragePooling2D(pool_size=(2, 2))(pool2_1)
    v_x_f_3 = Lambda(lambda b: concatenate([tf.slice(b[0], [0, 0, 0, int(i*256/2)], [-1, -1, -1, int(256/2)])*tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)], axis=-1))([pool3, pool3_1])
    conv4 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_x_f_3)
    conv4 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv4)

    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)
    pool4_1 = AveragePooling2D(pool_size=(2, 2))(pool3_1)
    v_x_f_4 = Lambda(lambda b: concatenate([tf.slice(b[0], [0, 0, 0, int(i*512/2)], [-1, -1, -1, int(512/2)])*tf.slice(b[1], [0, 0, 0, i], [-1, -1, -1, 1]) for i in range(2)], axis=-1))([pool4, pool4_1])

    conv5 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(v_x_f_4)
    conv5 = Conv2D(1024, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='tanh', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(48, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(4, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1)(conv9)
    out = Add()([conv10, input])
    model = Model(inputs=inputs, outputs=out, name='Unet_6')

    model.compile(optimizer=Adam(lr=1e-3), loss='mae', metrics=['mae'])

    print(model.summary())

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

#########################################################################################################################################
