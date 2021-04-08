# based on https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
# and https://github.com/ChunML/ssd-tf2/blob/master/network.py
import tensorflow as tf
import numpy as np
from dlf.layers.l2_normalization import L2Normalization


def VGG16_SSD(num_classes,
              ratios,
              weights='imagenet',
              input_shape=(512, 512, 3),
              ssd_300=True):
    """Creates a VGG16 based SSD network

    # Arguments
        num_classes: int. number of classes to train
        ratios: [float]. ratios for anchor boxes. len == 7
        weights: str. Weights for vgg base network. Defaults to 'imagenet'.
        input_shape: tuple. Dimension for input. Defaults to (512, 512, 3).
        ssd_300: bool. Is a ssd 300 or 512. Defaults to True.

    # Returns
        - tf.keras.models.Modell. SSD model
        - list[int]. List of feature map dimensions used for anchor box generation
    """
    shape = (300, 300, 3) if ssd_300 else input_shape

    num_anchors = [len(x) + 1 for x in ratios]
    img_input = tf.keras.layers.Input(shape=shape)
    conf_layers = create_conf_head_layers(num_classes, num_anchors)
    loc_layers = create_loc_head_layers(num_anchors)

    # Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block1_conv1')(img_input)
    x = tf.keras.layers.Conv2D(64, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block1_conv2')(x)
    x = tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block2_conv1')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block2_conv2')(x)
    x = tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)

    # Block 3
    x = tf.keras.layers.Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv1')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv2')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv3')(x)
    x = tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)

    # Block 4
    x = tf.keras.layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv3')(x)
    block4_conv3 = L2Normalization(
        gamma_init=20, name='block4_conv3_l2_norm')(x)

    x = tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding='same', name='block4_pool')(block4_conv3)

    # Block 5
    x = tf.keras.layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv3')(x)
    x = tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(1, 1),
        padding='same', name='block5_pool')(x)

    # FCN atrous conv2d
    fc1 = tf.keras.layers.Conv2D(1024, (3, 3),
                                 dilation_rate=6,
                                 activation='relu',
                                 padding='same',
                                 name='fc1')(x)

    fc2 = tf.keras.layers.Conv2D(1024, (1, 1),
                                 activation='relu',
                                 padding='same',
                                 name='fc2')(fc1)

    # Extra layers 8 to 11

    # Block 8
    x = tf.keras.layers.Conv2D(256, (1, 1),
                               activation='relu',
                               name='block8_conv1')(fc2)
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    block8_conv2 = tf.keras.layers.Conv2D(512, (3, 3),
                                          strides=2,
                                          activation='relu',
                                          name='block8_conv2')(x)

    # Block 9
    x = tf.keras.layers.Conv2D(128, (1, 1),
                               activation='relu',
                               name='block9_conv1')(block8_conv2)
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    block9_conv2 = tf.keras.layers.Conv2D(256, (3, 3),
                                          strides=2,
                                          activation='relu',
                                          name='block9_conv2')(x)

    # Block 10
    x = tf.keras.layers.Conv2D(128, (1, 1),
                               activation='relu',
                               name='block10_conv1')(block9_conv2)
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    block10_conv2 = tf.keras.layers.Conv2D(256, (3, 3),
                                           strides=2,
                                           activation='relu',
                                           name='block10_conv2')(x)

    # Block 11
    x = tf.keras.layers.Conv2D(128, (1, 1),
                               activation='relu',
                               name='block11_conv1')(block10_conv2)
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    block11_conv2 = tf.keras.layers.Conv2D(256, (3, 3),
                                           strides=2,
                                           activation='relu',
                                           name='block11_conv2')(x)

    skip_layers = [block4_conv3, fc2, block8_conv2, block9_conv2,
                   block10_conv2, block11_conv2]

    if not ssd_300:
        # Block 12
        x = tf.keras.layers.Conv2D(128, (1, 1),
                                   activation='relu',
                                   name='block12_conv1')(block11_conv2)
        x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
        block12_conv2 = tf.keras.layers.Conv2D(256, (4, 4),
                                               activation='relu',
                                               name='block12_conv2')(x)
        skip_layers.append(block12_conv2)

    confs = []
    for idx, skip_layer in enumerate(skip_layers):
        layer = conf_layers[idx]
        out = layer(skip_layer)
        out = tf.keras.layers.Reshape(
            [-1, num_classes])(out)
        confs.append(out)

    locs = []
    for idx, skip_layer in enumerate(skip_layers):
        layer = loc_layers[idx]
        out = layer(skip_layer)
        out = tf.keras.layers.Reshape(
            [-1, 4])(out)
        locs.append(out)

    confs = tf.keras.layers.Concatenate(name='concat_conf', axis=1)(confs)
    locs = tf.keras.layers.Concatenate(name='concat_loc', axis=1)(locs)

    model = tf.keras.models.Model(img_input, [confs, locs], name='vgg16')

    if weights == 'imagenet':
        init_pretrained_imagenet(model)

    feature_maps_sizes = []
    for layer in skip_layers:
        feature_maps_sizes.append(layer.shape[1])

    return model, feature_maps_sizes, tf.keras.applications.vgg16.preprocess_input


def init_pretrained_imagenet(model):
    vgg = tf.keras.applications.vgg16.VGG16(weights='imagenet')

    conv4_idx = vgg.layers.index(vgg.get_layer('block4_conv3'))
    for idx, layer in enumerate(vgg.layers[:conv4_idx]):
        weights = layer.get_weights()
        model.layers[idx].set_weights(weights)

    fc1_idx = vgg.layers.index(vgg.get_layer('fc1'))
    fc2_idx = vgg.layers.index(vgg.get_layer('fc2'))
    fc1_weights, fc1_biases = vgg.layers[fc1_idx].get_weights()
    fc2_weights, fc2_biases = vgg.layers[fc2_idx].get_weights()

    fc1_weights = np.random.choice(
        np.reshape(fc1_weights, (-1,)), (3, 3, 512, 1024))
    fc1_biases = np.random.choice(
        fc1_biases, (1024,))

    fc2_weights = np.random.choice(
        np.reshape(fc2_weights, (-1,)), (1, 1, 1024, 1024))
    fc2_biases = np.random.choice(
        fc2_biases, (1024,))

    model.get_layer('fc1').set_weights([fc1_weights, fc1_biases])
    model.get_layer('fc2').set_weights([fc2_weights, fc2_biases])


def create_conf_head_layers(num_classes, num_anchors):
    """ Create layers for classification
    """
    conf_head_layers = [
        tf.keras.layers.Conv2D(num_anchors[0] * num_classes, kernel_size=3,
                               padding='same', name='conf4_conv2d'),  # for 4th block
        tf.keras.layers.Conv2D(num_anchors[1] * num_classes, kernel_size=3,
                               padding='same', name='conf7_conv2d'),  # for 7th block
        tf.keras.layers.Conv2D(num_anchors[2] * num_classes, kernel_size=3,
                               padding='same', name='conf8_conv2d'),  # for 8th block
        tf.keras.layers.Conv2D(num_anchors[3] * num_classes, kernel_size=3,
                               padding='same', name='conf9_conv2d'),  # for 9th block
        tf.keras.layers.Conv2D(num_anchors[4] * num_classes, kernel_size=3,
                               padding='same', name='conf10_conv2d'),  # for 10th block
        tf.keras.layers.Conv2D(num_anchors[5] * num_classes, kernel_size=3,
                               padding='same', name='conf11_conv2d'),  # for 11th block
    ]

    if len(num_anchors) == 7:
        conf_head_layers.append(
            tf.keras.layers.Conv2D(num_anchors[6] * num_classes, kernel_size=1,
                                   name='conf12_conv2d')  # for 12th block
        )

    return conf_head_layers


def create_loc_head_layers(num_anchors):
    """ Create layers for regression
    """
    loc_head_layers = [
        tf.keras.layers.Conv2D(num_anchors[0] * 4, kernel_size=3,
                               padding='same', name='loc4_conv2d'),
        tf.keras.layers.Conv2D(num_anchors[1] * 4, kernel_size=3,
                               padding='same', name='loc7_conv2d'),
        tf.keras.layers.Conv2D(num_anchors[2] * 4, kernel_size=3,
                               padding='same', name='loc8_conv2d'),
        tf.keras.layers.Conv2D(num_anchors[3] * 4, kernel_size=3,
                               padding='same', name='loc9_conv2d'),
        tf.keras.layers.Conv2D(num_anchors[4] * 4, kernel_size=3,
                               padding='same', name='loc10_conv2d'),
        tf.keras.layers.Conv2D(num_anchors[5] * 4, kernel_size=3,
                               padding='same', name='loc11_conv2d'),
    ]

    if len(num_anchors) == 7:
        loc_head_layers.append(
            tf.keras.layers.Conv2D(num_anchors[6] * 4, kernel_size=1,
                                   name='loc12_conv2d')
        )

    return loc_head_layers
