from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3DTranspose
from keras.engine import Model
from keras.optimizers import Adam

#from .unet import create_convolution_block, concatenate
try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

from ..metrics import weighted_dice_coefficient_loss


create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def BrainNet_model(input_shape=(4, 128, 128, 128), n_base_filters=24, depth=6, dropout_rate=0.3,
                      n_hierarchical_levels=4, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    """
    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_hierarchical_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)
    current_layer = inputs
    level_output_layers = list()
    level_filters = list()

    for level_number in range(depth):
        n_level_filters = n_base_filters * (2**level_number)
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        local_skip_layer = local_skip_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, local_skip_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    hierarchical_skip_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, up_filters=current_layer._keras_shape[1], 
                                conv_filters=level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_convolution_block(input_layer, n_filters=level_filters[level_number])
        current_layer = localization_output
        if level_number < n_hierarchical_levels:
            hierarchical_skip_layers.insert(0, create_convolution_block(current_layer, n_filters=n_labels, kernel=(1, 1, 1)))

    output_layer = None
    for level_number in reversed(range(n_hierarchical_levels)):
        hierarchical_skip_layer = hierarchical_skip_layers[level_number]
        if output_layer is None:
            output_layer = hierarchical_skip_layer
        else:
            output_layer = Add()([output_layer, hierarchical_skip_layer])

        if level_number > 0:
            # output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)
            output_layer = get_up_convolution(n_filters=output_layer._keras_shape[1], kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=True):

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model



######### Helper Functions ############
def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):

    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def create_up_sampling_module(input_layer, up_filters, conv_filters, kernel_size=(2,2,2), strides=(2,2,2), 
                                pool_size=(2,2,2), deconvolution=True):
    if deconvolution:
        up_sample = Conv3DTranspose(filters=up_filters, kernel_size=kernel_size,
                               strides=strides)(input_layer)
    else:
        up_sample = UpSampling3D(size=pool_size)(input_layer)

    convolution = create_convolution_block(up_sample, conv_filters)
    return convolution

def get_up_convolution(n_filters, pool_size=(2,2,2), kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=True):
    if deconvolution:
        return Conv3DTranspose(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

def local_skip_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2


