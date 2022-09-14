import numpy as np
import keras
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(64, 64), shuffle=True, _max='level'):
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
            tracers = np.load('/home/jacob/PycharmProjects/DL_Advection_Emulation/Model_Applications_for_NOAA/Prepped_Data/Tracers/tracer_'+str(ID)+'.npy')
            time, z, slice, tracer, max = np.load('/home/jacob/PycharmProjects/DL_Advection_Emulation/Model_Applications_for_NOAA/Prepped_Data/Tracers/tracer_'+str(ID)+'_info.npy')
            u = np.load('/home/jacob/PycharmProjects/DL_Advection_Emulation/Model_Applications_for_NOAA/Prepped_Data/Winds/u_scaled_' + time + '_' + z + '_' + slice + '.npy')
            v = np.load('/home/jacob/PycharmProjects/DL_Advection_Emulation/Model_Applications_for_NOAA/Prepped_Data/Winds/v_scaled_' + time + '_' + z + '_' + slice + '.npy')
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

