import numpy as np
from tensorflow import keras
from keras.callbacks import *
import tensorflow as tf
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(64, 64), shuffle=True, training=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        if training == True:
            self.dir = '/home/jacob/PycharmProjects/DL_Advection_Emulation/Claw_data/training/'
        else:
            self.dir = '/home/jacob/PycharmProjects/DL_Advection_Emulation/Claw_data/validation/'

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
        x = np.empty((self.batch_size, *self.dim, 3))
        y = np.empty((self.batch_size, *self.dim, 1), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            data = np.load(self.dir + 'ex_'+str(ID)+'.npy')
            data[:, :, 1] = data[:, :, 1] / self.dim[0]
            data[:, :, 1] = data[:, :, 2] / self.dim[1]
            x[i] = data[:, :, :-1]
            y[i] = data[:, :, [-1]]
        return x, y

from DL_Models.Unet import Unet_base
from DL_Models.FCN import FCN_narrow
from DL_Models.FCN import FCN_straight
from DL_Models import AdvecBlock

def train(model, num_train, num_val, batch_size, epochs, directory, label):
    train_generator = DataGenerator(np.arange(num_train)[1:], dim=(64, 64), batch_size=batch_size, training=True)
    val_generator = DataGenerator(np.arange(num_val), dim=(64, 64), batch_size=batch_size, training=False)
    print(label)
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),
                 EarlyStopping(monitor='val_loss', patience=10),
                 TensorBoard(log_dir=directory + label + '_logs', update_freq='epoch')]
    # Train model on dataset
    print('training')
    model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=epochs, callbacks=callbacks)
    print('saving')
    model.save(directory + label)



def eval(models, labels, directory='/home/jacob/PycharmProjects/DL_Advection_Emulation/Claw_data'):
    from sklearn.metrics import mean_absolute_error as mae
    import matplotlib.pyplot as plt
    #we will collect errors for each model as well as errors for the numerical method at 1x, 2x->1x, and 4x->1x resolution
    for res in [32, 64, 128]:
        print(res)
        if res == 32:
            v, u = np.mgrid[0:1:32j, 0:1:32j]
        elif res == 64:
            v, u = np.mgrid[0:1:64j, 0:1:64j]
        else:
            v, u = np.mgrid[0:1:128j, 0:1:128j]
        errors = np.zeros((255, len(models) + 4))
        plot_steps = [0, 63, 127, 255]
        saved_solns = np.zeros((len(plot_steps), len(models) + 1, res, res))
        last_soln = np.zeros((res, res))
        for i, model in enumerate(models):
            m1 = model
            #m1 = model(input_shape=(res, res, 3))
            #m1.set_weights(m1_build.get_weights())
            for step in range(255):
                if step==0:
                    data_inp = np.load(directory + '/testing/' + str(int(res * 8)) + '->' + str(res) + '_step:' + str(0) + '.npy')
                    print('inp shape is: ', data_inp.shape)
                else:
                    data_inp = last_soln
                velocity_u = np.sin(np.pi * u) ** 2 * np.sin(2 * np.pi * v) * np.cos(np.pi * dt * step / tfinal)
                velocity_v = np.sin(np.pi * v) ** 2 * np.sin(2 * np.pi * u) * np.cos(np.pi * dt * step / tfinal)
                ref_soln = np.load(directory + '/testing/' + str(int(res * 8)) + '->' + str(res) + '_step:' + str(step + 1) + '.npy')
                last_soln = m1(np.concatenate([np.expand_dims(data_inp, axis=(0, -1)), np.expand_dims(velocity_u, axis=(0, -1)), np.expand_dims(velocity_v, axis=(0, -1))], axis=-1))[0, :, :, 0]
                if step in plot_steps:
                    saved_solns[plot_steps.index(step), -1] = ref_soln
                    saved_solns[plot_steps.index(step), i] = last_soln
                    errors[step, i] = mae(saved_solns[plot_steps.index(step), i], ref_soln)
                else:
                    errors[step, i] = mae(ref_soln, last_soln)
                errors[step, -1] = mae(np.load(directory + '/testing/' + str(int(res * 4)) + '->' + str(res) + '_step:' + str(step + 1) + '.npy'), ref_soln)
                errors[step, -2] = mae(np.load(directory + '/testing/' + str(int(res * 2)) + '->' + str(res) + '_step:' + str(step + 1) + '.npy'), ref_soln)
                errors[step, -3] = mae(np.load(directory + '/testing/' + str(int(res * 1)) + '->' + str(res) + '_step:' + str(step + 1) + '.npy'), ref_soln)
                errors[step, -4] = np.mean(np.abs(ref_soln))
        x = np.arange(255)
        for i in range(len(models)):
            plt.plot(x, errors[:, i], label=labels[i], color='red')
        plt.plot(x, errors[:, -4], label='Avg abs soln value, res=' + str(res) + 'x' + str(res), color='black')
        plt.plot(x, errors[:, -3], label='1x res FV soln', color='blue')
        plt.plot(x, errors[:, -2], label='2x res FV soln', color='green')
        plt.plot(x, errors[:, -1], label='4x res FV soln', color='orange')
        plt.legend()
        plt.title(str(res) + 'x' + str(res) + ' MAEs')
        plt.xlabel('Simulation Step')
        plt.ylabel('MAE')
        plt.show()
        xlist = np.linspace(0, 1, res)
        ylist = np.linspace(0, 1, res)
        X, Y = np.meshgrid(xlist, ylist)
        for j in range(len(plot_steps)):
            for z in range(len(models)):
                fig, ax = plt.subplots(1, 1)
                cp = ax.contourf(X, Y, saved_solns[j, z], levels=10)
                fig.colorbar(cp)  # Add a colorbar to a plot
                ax.set_title(str(res) + 'x' + str(res) + 'Solution, ' + str(z) + 'th model, ' + 'step=' + str(plot_steps[j]))
                ax.set_xlabel('x axis')
                ax.set_ylabel('y axis')
                plt.show()
            fig, ax = plt.subplots(1, 1)
            cp = ax.contourf(X, Y, saved_solns[j, -1], levels=10)
            fig.colorbar(cp)  # Add a colorbar to a plot
            ax.set_title(str(res) + 'x' + str(res) + 'Ref Solution, ' + 'step=' + str(plot_steps[j]))
            ax.set_xlabel('x axis')
            ax.set_ylabel('y axis')
            plt.show()

num_train = 20000
num_train = 2000
#num_val = 5000
num_val = 300
epochs = 1
batch_size = 8
dt = 1/256
tfinal = 1.0
directory = '/home/jacob/PycharmProjects/DL_Advection_Emulation/DL_Models/'

# base network
label = 'advection block 1.1'
#label = 'unet 1.3'
model1 = AdvecBlock.Numerical_model(AdvecBlock.AdvecBlock4, square=True, mult_wind=False, stack_wind=True, padding_offset=-1, smooth_padd=False, flip_indicators=True)
model1.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss='mae', metrics=['mae'])
#model1 = Unet_base.model6
#m1_build = model1()
#model1(tf.zeros((1, 64, 64, 3)))
#print(m1_build.summary())
#model1(AdvecBlock.AdvecBlock4, input_size=(64, 64, 3), square=True, mult_wind=False, stack_wind=True, padding_offset=-1, smooth_padd=False, flip_indicators=True)
print('starting...')
print('model 1 built')
#train(m1_build, num_train, num_val, batch_size, epochs, directory, label)
train(model1, num_train, num_val, batch_size, epochs, directory, label)
print('done model 1')
eval([model1], labels=[label])
exit()
label = 'fcn_n_5'
model1 = FCN_narrow.model5()
train(model1, num_train, num_val, batch_size, epochs, directory, label)
label = 'fcn_n_6'
model1 = FCN_narrow.model6()
train(model1, num_train, num_val, batch_size, epochs, directory, label)
label = 'unet_4'
model1 = Unet_base.model4()
train(model1, num_train, num_val, batch_size, epochs, directory, label)
label = 'unet_6'
model1 = Unet_base.model6()
train(model1, num_train, num_val, batch_size, epochs, directory, label)
exit()

unet1 = Unet_base.model4()
unet2 = Unet_base.model5()
unet3 = Unet_base.model6()

fcn_n1 = FCN_narrow.model4()
fcn_n2 = FCN_narrow.model5()
fcn_n3 = FCN_narrow.model6()

fcn_s1 = FCN_straight.model4()
fcn_s2 = FCN_straight.model5()
fcn_s3 = FCN_straight.model6()
# try not squaring smoothness indicators
#adv2 = AdvecBlock.Numerical_model(block=AdvecBlock.AdvecBlock2, input_size=(256, 256, 3), square=False, mult_wind=False, stack_wind=False, padding_offset=0, extra_conv=False, smooth_padd=False, flip_indicators=True, num_derivs=1)

# try not flipping indicators when creating weights
#adv3 = AdvecBlock.Numerical_model(block=AdvecBlock.AdvecBlock2, input_size=(256, 256, 3), square=True, mult_wind=False, stack_wind=False, padding_offset=0, extra_conv=False, smooth_padd=False, flip_indicators=False, num_derivs=1)

# try the offset padding
#adv4 = AdvecBlock.Numerical_model(block=AdvecBlock.AdvecBlock2, input_size=(256, 256, 3), square=True, mult_wind=False, stack_wind=False, padding_offset=-1, extra_conv=False, smooth_padd=False, flip_indicators=True, num_derivs=1)

# try multiplying indicators by velocity
#adv5 = AdvecBlock.Numerical_model(block=AdvecBlock.AdvecBlock2, input_size=(256, 256, 3), square=True, mult_wind=True, stack_wind=False, padding_offset=0, extra_conv=False, smooth_padd=False, flip_indicators=True, num_derivs=1)

# try an additional higher order spatial derivative estimate
#adv6 = AdvecBlock.Numerical_model(block=AdvecBlock.AdvecBlock2, input_size=(256, 256, 3), square=True, mult_wind=False, stack_wind=False, padding_offset=0, extra_conv=False, smooth_padd=False, flip_indicators=True, num_derivs=2)

#models = [adv1, adv2, adv3, adv4, adv5, adv6, unet1, unet2, unet3, fcn_n1, fcn_n2, fcn_n3, fcn_s1, fcn_s2, fcn_s3]
labels = ['adv1', 'adv2', 'unet1', 'unet2', 'unet3', 'fcn_n1', 'fcn_n2', 'fcn_n3', 'fcn_s1', 'fcn_s2', 'fcn_s3']
num_train = 20000
num_val = 5000
epochs = 10
batch_size = 32
directory = '/home/jacob/PycharmProjects/DL_Advection_Emulation/DL_Models/'







