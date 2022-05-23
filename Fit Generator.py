import numpy as np
import keras
import tensorflow as tf
from Unet import Unet_base as unet
from Autoencoder import Autoencoder_base as auto_encoder
from FCN import FCN_narrow as fcn_n
from FCN import FCN_widen as fcn_w
from FCN import FCN_straight as fcn_s
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(64, 64), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, *self.dim, 5))
        y = np.empty((self.batch_size, *self.dim, 1), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            tracers = np.load('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/tracer_'+str(ID)+'.npy')
            time, z, slice, tracer, max = np.load('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/tracer_'+str(ID)+'_info.npy')
            u = np.load('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Winds/u_scaled_' + time + '_' + z + '_' + slice + '.npy')
            v = np.load('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Winds/v_scaled_' + time + '_' + z + '_' + slice + '.npy')
            x[i, :, :, 0] = tracers[:, :, 0]
            x[i, :, :, 1] = u[:-1, :]
            x[i, :, :, 2] = u[1:, :]
            x[i, :, :, 3] = v[:, :-1]
            x[i, :, :, 4] = v[:, 1:]
            # Store class
            y[i] = np.expand_dims(tracers[:, :, -1], axis=-1)
            #print(np.min(x), np.max(x), np.mean(x))
            #print(np.min(y), np.max(y), np.mean(y), max)
            #print(tracer)
            #print("**************************")

        return x, y

########################################################################################

from keras.models import Sequential

# Parameters
epochs = 1
params = {'dim': (64,64),
          'batch_size': 32,
          'shuffle': True}

# Datasets
partition_train = np.load('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/train_indx.npy')
partition_val = np.load('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/val_indx.npy')
# Generators
training_generator = DataGenerator(partition_train, **params)
validation_generator = DataGenerator(partition_val, **params)
#callbacks
directory_base = '/home/jacob/PycharmProjects/DL Advection Emulation/DL Models/'
directory = ''

#********************************************************************************************************************

name = 'unet4'
model = unet.model4()
directory = directory_base + 'Unet/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************
name = 'unet5'
model = unet.model5()
directory = directory_base + 'Unet/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************
name = 'unet6'
model = unet.model6()
directory = directory_base + 'Unet/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************

#====================================================================================================================

#********************************************************************************************************************
name = 'ae4'
model = auto_encoder.model4()
directory = directory_base + 'Autoencoder/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************
name = 'ae5'
model = auto_encoder.model5()
directory = directory_base + 'Autoencoder/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************
name = 'ae6'
model = auto_encoder.model6()
directory = directory_base + 'Autoencoder/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************

#====================================================================================================================

#********************************************************************************************************************
name = 'fcn_n4'
model = fcn_n.model4()
directory = directory_base + 'FCN/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************
name = 'fcn_n5'
model = fcn_n.model5()
directory = directory_base + 'FCN/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************
name = 'fcn_n6'
model = fcn_n.model6()
directory = directory_base + 'FCN/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************

#====================================================================================================================

#********************************************************************************************************************
name = 'fcn_w4'
model = fcn_w.model4()
directory = directory_base + 'FCN/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************
name = 'fcn_w5'
model = fcn_w.model5()
directory = directory_base + 'FCN/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************
name = 'fcn_w6'
model = fcn_w.model6()
directory = directory_base + 'FCN/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************

#====================================================================================================================

#********************************************************************************************************************
name = 'fcn_s4'
model = fcn_n.model4
directory = directory_base + 'FCN/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************
name = 'fcn_s5'
model = fcn_n.model5()
directory = directory_base + 'FCN/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************
name = 'fcn_s6'
model = fcn_n.model6()
directory = directory_base + 'FCN/' + name
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7),
             EarlyStopping(monitor='val_loss', patience=15),
             TensorBoard(log_dir=directory+'_logs', update_freq='epoch')]
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks)
model.save(directory)
#********************************************************************************************************************






















#model = keras.models.load_model('path/to/location')
exit()
partition_train = np.load('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/train_indx.npy')
partition_val = np.load('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/val_indx.npy')
partition_test = np.load('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/test_indx.npy')
print(len(partition_test), len(partition_val), len(partition_train), len(partition_test) + len(partition_val) + len(partition_train))
maxes = []
maxes1 = []
for i in partition_train:
    tracers = np.load('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/tracer_' + str(i) + '.npy')
    max_i = np.max(tracers)
    maxes.append(max_i)
    maxes1.append(np.expand_dims(tracers[:, :, -1], axis=-1))
    print(max_i)
print(np.mean(maxes), np.std(maxes), np.min(maxes), np.max(maxes))
_ = plt.hist(maxes, bins='auto')
plt.show()
_1 = plt.hist(maxes, bins='auto')
plt.show()
exit()
#********************************************************************************************************************
