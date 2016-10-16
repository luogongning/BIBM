from __future__ import print_function

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras import backend as K
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
from keras.utils.visualize_util import plot


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)
	
def get_model_ALL_ED():
    model = Sequential()
    model.add(Activation(activation=center_normalize, input_shape=(3, 92, 92)))

    model.add(Convolution2D(16, 31, 31, border_mode='valid'))
    model.add(Activation('sigmoid'))
    #model.add(Convolution2D(64, 7, 7, border_mode='valid'))
    #model.add(Activation('relu'))
#    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(64, 6, 6, border_mode='valid'))
    model.add(Activation('relu'))
#    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    #model.add(Convolution2D(96, 5, 5, border_mode='valid'))
    #model.add(Activation('relu'))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
#    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
    #model.add(Convolution2D(128, 3, 3, border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, W_regularizer=l2(1e-3)))
    model.add(Activation('relu'))
    model.add(Dense(256, W_regularizer=l2(1e-4)))
    model.add(Activation('relu'))
    model.add(Dense(128, W_regularizer=l2(1e-4)))
    
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('relu'))

    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=root_mean_squared_error)
    return model

model_diastole = get_model_ALL_ED()
#SVG(model_to_dot(model_diastole).create(prog='dot', format='svg'))
plot(model_diastole, to_file='model_systole2.png', show_shapes=True)