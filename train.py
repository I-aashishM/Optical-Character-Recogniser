
import string
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda,Bidirectional
from keras.models import Model, Sequential, load_model
import keras.backend as K
from keras.callbacks import ModelCheckpoint


char_list = string.ascii_letters + string.digits
def model_ocr():


    inputs = Input(shape=(32, 128, 1))

    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)  # convolution layer 1
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)  # poolig layer with kernel size (2,2)

    conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_1)  # convolution layer 2
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_2)  # convolution layer 3
    batch_norm_1 = BatchNormalization()(conv_3)

    conv_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch_norm_1)  # convolution layer 4
    batch_norm_2 = BatchNormalization()(conv_4)  # Batch normalization layer

    conv_5 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch_norm_2)  # convolution layer 5
    batch_norm_3 = BatchNormalization()(conv_5)  # Batch normalization layer
    pool_3 = MaxPool2D(pool_size=(3, 1))(batch_norm_3)

    conv_6 = Conv2D(64, (2, 2), activation='relu')(pool_3)  # convolution layer 6

    squeezed_layer = Lambda(lambda x: K.squeeze(x, 1))(conv_6)

    # bidirectional LSTM layers with units = 128 and 64
    blstm_layer_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed_layer)
    blstm_layer_2 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(blstm_layer_1)

    outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_layer_2)

    act_model = Model(inputs, outputs)
    return  act_model