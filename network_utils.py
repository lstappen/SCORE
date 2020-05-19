import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, Flatten
from keras.layers import Conv2D, Conv1D, MaxPooling2D, GlobalAveragePooling2D, Reshape, LSTM, Lambda, concatenate, MaxPooling1D, LocallyConnected1D, Bidirectional
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
import os
from utils import parameter_path

# Includes some models which did not make it into the paper.

# Plot the training and the loss graph
def visualise_training(config, history, fold, parameter):
    # summarize history for accuracy
    path = parameter_path(config['PLOT_PATH'], parameter)

    print("epochs: ", len(history.history['val_loss']))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(path, 'fold_' + str(fold) + '_accuracy.png'))
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(path, 'fold_' + str(fold) + '_loss.png'))
    plt.close()

# create the general model part
def general_model_part(model, learning_rate):
    # Opt
    opt = Adam(learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    # Compile the model
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    # Display model architecture summary
    print(model.summary())

    return model

# create the cnn model
def create_cnn_model(learning_rate, input_shape, num_labels):

    # Construct model
    # batch_normalisation
    model = Sequential()
    #1st layer
    model.add(Conv2D(filters=16, kernel_size=2
                     , activation='relu'
                     , input_shape=input_shape
                     , name='Conv2D1'))  #
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.2))

    #2nd layer
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu', name='Conv2D2'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.2))

    #3rd layer
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu', name='Conv2D3'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.2))

    #4th layer
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu', name='Conv2D4'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.2))
    model.add(GlobalAveragePooling2D(name='main_input'))

    #output layer
    model.add(Dense(num_labels, activation='softmax', name='output'))

    return general_model_part(model, learning_rate)

# create the cnn model
def create_lstm_model(config, learning_rate, input_shape, num_labels):

    # Construct model
    # batch_normalisation
    inputs = Input(input_shape)
    model = Bidirectional(LSTM(config['lstm1_n'], return_sequences=True))(inputs)
    model = Dropout(rate=0.5)(model)
    model = Bidirectional(LSTM(config['lstm2_n'], return_sequences=False))(model)
    model = Dropout(rate=0.5)(model)

    #output layer
    model = Dense(num_labels, activation='softmax', name='output')(model)
    lstm = Model(inputs=[inputs], outputs=model)

    return general_model_part(lstm, learning_rate)

# create the cnn model
def create_crnn_model(learning_rate, input_shape, num_labels):

    # Construct model
    inputs = Input(input_shape)
    #1st layer
    model = Conv2D(filters=16, kernel_size=2
                     , name='Conv2D1')(inputs)  
    model = Activation('relu')(model)    
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=2)(model)
    model = Dropout(rate=0.2)(model)

    #2nd layer
    model = Conv2D(filters=32, kernel_size=2, name='Conv2D2')(model)
    model = Activation('relu')(model)    
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=2)(model)
    model = Dropout(rate=0.2)(model)

    #3rd layer
    model = Conv2D(filters=64, kernel_size=2, name='Conv2D3')(model)
    model = Activation('relu')(model)    
    model = BatchNormalization()(model)    
    model = MaxPooling2D(pool_size=2)(model)
    model = Dropout(rate=0.2)(model)

    #4th layer
    model = Conv2D(filters=128, kernel_size=2, name='Conv2D4')(model)
    model = Activation('relu')(model)    
    model = BatchNormalization()(model) # (None, 3, 3, 128)

    # CNN to RNN
    model = Reshape(target_shape=((9, 128)), name='reshape')(model)  # (None, 32, 2048)
    inner = Dense(64, kernel_initializer='he_normal', name='dense1')(model)  # (None, 32, 64)

    # RNN layer
    lstm_1 = LSTM(64, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 32, 512)
    lstm_1b = LSTM(64, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)

    lstm1_merged = concatenate([lstm_1, reversed_lstm_1b])  # (None, 32, 512)
    lstm1_merged = BatchNormalization()(lstm1_merged)
    
    lstm_2 = LSTM(64, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    model = BatchNormalization()(lstm_2)

    #output layer
    model = Dense(num_labels, activation='softmax', name='output')(model)
    crnn = Model(inputs=[inputs], outputs=model)

    return general_model_part(crnn, learning_rate)



# create the cnn model
def create_crnn_small_model(learning_rate, input_shape, num_labels):

    # Construct model
    inputs = Input(input_shape)
    #1st layer
    model = Conv2D(filters=40, kernel_size=4
                     , name='Conv2D1')(inputs)  
    model = Activation('relu')(model)    
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=6)(model)
    model = Dropout(rate=0.2)(model)

    #2nd layer
    model = Conv2D(filters=128, kernel_size=3, name='Conv2D2')(model)
    model = Activation('relu')(model)    
    model = BatchNormalization()(model)
    model = MaxPooling2D(pool_size=(4,1))(model)
    #model = Dropout(rate=0.2)(model)

    test = Model(inputs=[inputs], outputs=model)
    general_model_part(test, learning_rate)

    # CNN to RNN
    model = Reshape(target_shape=((9, 128)), name='reshape')(model)  # (None, 32, 2048)
    inner = Dense(64, kernel_initializer='he_normal', name='dense1')(model)  # (None, 32, 64)

    # RNN layer
    lstm_1 = LSTM(64, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 32, 512)
    lstm_1b = LSTM(64, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)

    lstm1_merged = concatenate([lstm_1, reversed_lstm_1b])  # (None, 32, 512)
    lstm1_merged = BatchNormalization()(lstm1_merged)
    
    lstm_2 = LSTM(64, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    model = BatchNormalization()(lstm_2)

    #output layer
    model = Dense(num_labels, activation='softmax', name='output')(model)
    crnn = Model(inputs=[inputs], outputs=model)

    return general_model_part(crnn, learning_rate)

def create_cnn_end2end_model(config, learning_rate, input_shape, num_labels):
    # Construct model
    inputs = Input(input_shape)
    #1st layer 
    model = Conv2D(config['Conv1_filters'], config['Conv1_kernel_size'], padding='same')(inputs) 
    model = Activation('relu')(model) 
    model = MaxPooling2D(pool_size = (2), strides = (2), padding = 'same')(model)
    model = Dropout(rate=0.5)(model) 
    
    # #2nd layer
    if config['Conv2_filters'] != 0:
        model = Conv2D(config['Conv2_filters'], config['Conv2_kernel_size'],padding='same')(model) 
        model = Activation('relu')(model) 
        model = MaxPooling2D(pool_size = (2), strides = (2), padding = 'same')(model)
        model = Dropout(rate=0.5)(model) 

    if config['Conv3_filters'] != 0:
        #2nd layer
        model = Conv2D(config['Conv3_filters'], config['Conv3_kernel_size'],padding='same')(model) 
        model = Activation('relu')(model) 

    model = Bidirectional(LSTM(config['lstm1_n'], return_sequences=True))(model)
    #model = Dropout(rate=0.5)(model)
    model = Bidirectional(LSTM(config['lstm2_n'], return_sequences=False))(model)
    #output layer
    model = Dense(num_labels, activation='softmax', name='output')(model)
    cnn = Model(inputs=[inputs], outputs=model)

    return general_model_part(cnn, learning_rate)

# create the cnn model
def create_crnn_end2end_model(config, learning_rate, input_shape, num_labels):

    # Construct model
    inputs = Input(input_shape)
    #1st layer 
    model = Conv1D(config['Conv1_filters'], config['Conv1_kernel_size'], padding='same')(inputs) 
    model = Activation('relu')(model) 
    model = MaxPooling1D(pool_size = (2), strides = (2), padding = 'same')(model)
    model = Dropout(rate=0.5)(model) 
    
    # #2nd layer
    if config['Conv2_filters'] != 0:
        model = Conv1D(config['Conv2_filters'], config['Conv2_kernel_size'],padding='same')(model) 
        model = Activation('relu')(model) 
        model = MaxPooling1D(pool_size = (2), strides = (2), padding = 'same')(model)
        model = Dropout(rate=0.5)(model) 

    if config['Conv3_filters'] != 0:
        #2nd layer
        model = Conv1D(config['Conv3_filters'], config['Conv3_kernel_size'],padding='same')(model) 
        model = Activation('relu')(model) 

    # RNN layer
    model = Bidirectional(LSTM(config['lstm1_n'], return_sequences=True))(model)
    model = Bidirectional(LSTM(config['lstm2_n'], return_sequences=False))(model)

    #output layer
    model = Dense(num_labels, activation='softmax', name='output')(model)
    crnn = Model(inputs=[inputs], outputs=model)

    return general_model_part(crnn, learning_rate)