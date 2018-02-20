from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, ZeroPadding1D,
                          TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)
from keras.activations import relu, elu
from keras.layers.advanced_activations import LeakyReLU

def clipped_relu(x):
    return relu(x, max_value=20)

# model 0
def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model 1
def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    gru = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(gru)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model 2
def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1, layers=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None

    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)

    output_length = input_length
    if border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1

    # TODO: wrong for multiple layers
    for l in range(layers):
        output_length = (output_length + stride - 1) // stride

    return output_length

# model 3
def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    prev_input = input_data
    for i in range(recur_layers):
        gru = GRU(units, activation='elu', return_sequences=True, implementation=2, name='rnn'+str(i))(prev_input)
        bn_gru = (BatchNormalization(name='bn_rnn' + str(i))(gru))
        prev_input = bn_gru

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(prev_input)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model 4
def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='elu', return_sequences=True, implementation=2))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model 5
def more_units_model(input_dim, activation, units=512, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add recurrent layer
    o = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    o = BatchNormalization()(o)

    o = TimeDistributed(Dense(output_dim))(o)
    o = Activation('softmax', name='softmax')(o)
    # Specify the model
    model = Model(inputs=input_data, outputs=o)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model 6
def more_dense_model(input_dim, activation, units=256, output_dim=29):
    """
    1x GRU + 2x TDD
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    o = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    o = BatchNormalization()(o)
    # First TDD
    o = TimeDistributed(Dense(units))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    # Second TDD
    o = TimeDistributed(Dense(output_dim))(o)
    o = Activation('softmax', name='softmax')(o)
    # Specify the model
    model = Model(inputs=input_data, outputs=o)
    model.output_length = lambda x: x
    print(model.summary())
    return model


# model 7
def early_dense_model(input_dim, activation, units=256, output_dim=29):
    """
    2x TDD + 1x GRU + 1x Time Distributed Dense
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # First TDD
    o = TimeDistributed(Dense(units))(input_data)
    o = TimeDistributed(Activation(elu))(o)
    o = Dropout(0.5)(o)
    # Second TDD
    o = TimeDistributed(Dense(units))(o)
    o = TimeDistributed(Activation(elu))(o)
    o = Dropout(0.5)(o)

    # Add recurrent layer
    o = GRU(units, activation=activation,
            return_sequences=True, implementation=2, name='rnn')(o)

    # Final TDD
    o = TimeDistributed(Dense(output_dim))(o)
    o = Activation('softmax', name='softmax')(o)
    # Specify the model
    model = Model(inputs=input_data, outputs=o)
    model.output_length = lambda x: x
    print(model.summary())
    return model


# model 8
def more_bidir_model(input_dim, units=256, output_dim=29):
    """
    2x Bidir GRU instead of 1x Bidir GRU
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    o = Bidirectional(GRU(units, activation='elu', return_sequences=True, implementation=2))(input_data)
    o = Bidirectional(GRU(units, activation='elu', return_sequences=True, implementation=2))(o)

    o = TimeDistributed(Dense(output_dim))(o)
    o = Activation('softmax', name='softmax')(o)
    # Specify the model
    model = Model(inputs=input_data, outputs=o)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model 9
def more_cnn_model(input_dim, filters1=200, filters2=200, kernel_size=11, units=200, output_dim=29, dropout=0.2):
    """ Build a deep network for speech
    """
    conv_border_mode = 'same'
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    o = BatchNormalization()(input_data)
    o = Conv1D(filters1, kernel_size, strides=1, padding=conv_border_mode, activation=clipped_relu)(o)
    o = BatchNormalization()(o)
    o = Conv1D(filters2, kernel_size, strides=1, padding=conv_border_mode, activation=clipped_relu)(o)
    #o = LeakyReLU(alpha=.001, max_value=20)(o)
    o = Dropout(dropout)(o)
    o = BatchNormalization()(o)

    # Add a recurrent layer
    o = SimpleRNN(units, dropout=dropout,
                  return_sequences=True, implementation=2, name='rnn', activation=clipped_relu)(o)
    o = BatchNormalization(name='bn_rnn')(o)

    # and finally we add a TimeDistributed
    o = TimeDistributed(Dense(output_dim))(o)
    o = Activation('softmax', name='softmax')(o)
    # Specify the model
    model = Model(inputs=input_data, outputs=o)
    model.output_length = lambda x: x
    # model.output_length = lambda x: cnn_output_length( x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


# model 10
def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, fc=1024, units=512, dropout=0.3, output_dim=29):
    """
    Build a deep network for speech
    1dconv + 2 bidir layers + 2 tdd
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    o = BatchNormalization()(input_data)

    # Add a conv layer
    o = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='elu',
                     name='conv')(o)

    # Add 2 bidirectional recurrent layers
    o = Bidirectional(GRU(units, activation='elu', dropout=dropout, return_sequences=True, implementation=2))(o)
    o = Bidirectional(GRU(units, activation='elu', dropout=dropout, return_sequences=True, implementation=2))(o)
    o = BatchNormalization()(o)

    # Add 2 TimeDistributed layers
    o = TimeDistributed(Dense(fc))(o)
    o = TimeDistributed(Activation('elu'))(o)
    o = TimeDistributed(Dropout(dropout))(o)
    o = TimeDistributed(Dense(output_dim))(o)
    o = Activation('softmax', name='softmax')(o)

    # Specify the model
    model = Model(inputs=input_data, outputs=o)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


# TODO: more models to experiment with

# model 11
def no_rnn_model(input_dim, filters=128, layers=3, conv_stride=2, kernel_size=11, output_dim=29, dropout=0.2):
    """ Build a deep network for speech
    """
    conv_border_mode = 'same'
    # Main acoustic input

    input_data = Input(name='the_input', shape=(None, input_dim))
    o = BatchNormalization()(input_data)

    for l in range(layers):
        o = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation=clipped_relu)(o)
        filters *= 2

    o = BatchNormalization()(o)

    # add 2 TimeDistributed Dense layers
    o = TimeDistributed(Dense(1024))(o)
    o = TimeDistributed(Activation(clipped_relu))(o)
    o = Dropout(dropout)(o)
    o = TimeDistributed(Dense(output_dim))(o)
    o = Activation('softmax', name='softmax')(o)
    # Specify the model
    model = Model(inputs=input_data, outputs=o)
    # model.output_length = lambda x: x
    model.output_length = lambda x: cnn_output_length( x, kernel_size, conv_border_mode, conv_stride, layers=layers)
    print(model.summary())
    return model


# model 12
def rnn_language_model(input_dim, activation, units=256, output_dim=29):
    """
    1x RNN + Language Model + Time Distributed Dense?
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    o = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    o = BatchNormalization()(o)

    o = TimeDistributed(Dense(output_dim))(o)
    o = Activation('softmax', name='softmax')(o)
    # Specify the model
    model = Model(inputs=input_data, outputs=o)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# model 13
def model_2DConv(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, dropout=0.5, output_dim=29):
    """
    deep 2d conv. no time dense, no ctc
    tf kaggle challenge
    """
    pass

# model 14
def model_mozilla(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, dropout=0.5, output_dim=29):
    """
    Similar to Mozillas open source DeepSpeech
    mozilla  https://github.com/mozilla/DeepSpeech
    """
    pass

# model 15
def model_baidu(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, dropout=0.5, output_dim=29):
    """
    Similar to Baidu´s open source DeepSpeech
    baidu  https://github.com/baidu-research/ba-dls-deepspeech
    """
    pass

# model 16
def model_wavenet(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, dropout=0.5, output_dim=29):
    """
    Similar to Google´s Wavenet
    wavenet  https://github.com/buriburisuri/speech-to-text-wavenet
    """
    pass

# stacked vgg16
